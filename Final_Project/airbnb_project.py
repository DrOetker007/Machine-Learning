"""Reusable preparation and modeling helpers for the Airbnb price project."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor, Pool
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)


RANDOM_STATE = 42

SCORE_COLUMNS = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]

REVIEW_COUNT_COLUMNS = [
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
]

AMENITY_KEYWORDS = {
    "amenity_wifi": ["wifi"],
    "amenity_kitchen": ["kitchen"],
    "amenity_washer": ["washer"],
    "amenity_dryer": ["dryer"],
    "amenity_air_conditioning": ["air conditioning"],
    "amenity_workspace": ["workspace"],
    "amenity_parking": ["parking"],
    "amenity_elevator": ["elevator"],
    "amenity_dishwasher": ["dishwasher"],
    "amenity_oven": ["oven"],
    "amenity_coffee_maker": ["coffee maker"],
    "amenity_crib": ["crib"],
    "amenity_high_chair": ["high chair"],
    "amenity_outdoor_space": ["outdoor"],
    "amenity_city_view": ["city skyline view"],
    "amenity_hot_tub": ["hot tub"],
    "amenity_gym": ["gym"],
    "amenity_pool": ["pool"],
    "amenity_smoke_alarm": ["smoke alarm"],
    "amenity_bedroom_lock": ["lock on bedroom door"],
    "amenity_tv": ["tv"],
}

AMENITY_COLUMNS = list(AMENITY_KEYWORDS)
REVIEW_BUCKET_COLUMNS = [f"{column}_bucket" for column in SCORE_COLUMNS]

LINEAR_CATEGORICAL_COLUMNS = [
    "host_is_superhost",
    "neighbourhood_cleansed",
    "property_type",
    "property_type_grouped",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "bathroom_privacy",
    "bathroom_half_bath",
    "minimum_nights_bucket",
    "host_response_time",
    "instant_bookable",
] + REVIEW_BUCKET_COLUMNS + AMENITY_COLUMNS

LINEAR_LOG_NUMERIC_COLUMNS = REVIEW_COUNT_COLUMNS + [
    "minimum_nights",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_shared_rooms",
]

LINEAR_PLAIN_NUMERIC_COLUMNS = [
    "host_acceptance_rate_numeric",
    "latitude",
    "longitude",
    "bathrooms_per_guest",
]

CATBOOST_CATEGORICAL_COLUMNS = [
    "host_is_superhost",
    "neighbourhood_cleansed",
    "property_type",
    "property_type_grouped",
    "room_type",
    "bathrooms_text",
    "bathroom_privacy",
    "bathroom_half_bath",
    "host_response_time",
    "instant_bookable",
] + AMENITY_COLUMNS

CATBOOST_NUMERIC_COLUMNS = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "latitude",
    "longitude",
    "minimum_nights",
    "host_acceptance_rate_numeric",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_shared_rooms",
    "bathrooms_per_guest",
] + REVIEW_COUNT_COLUMNS + SCORE_COLUMNS

TEXT_COLUMNS = ["listing_text", "reviews_text", "amenities_text"]
TEXT_NUMERIC_COLUMNS = [
    "review_text_count",
    "listing_text_chars",
    "reviews_text_chars",
]

IMAGE_PROMPTS = [
    "a photo of a stylish apartment interior",
    "a photo of a cheap basic accommodation",
    "a photo of a modern apartment",
    "a photo of old furniture",
    "a photo of a terrace",
    "a photo of a double bed",
    "a photo of a washing machine in an apartment",
    "a photo of a Japanese style room",
    "a photo of a tatami room",
    "a photo of a cluttered room",
    "a photo of a small basic room",
    "a photo of an old bathroom",
    "a photo with text on it",
    "a photo of multiple pictures in one image",
    "a photo of a collage",
]


def parse_price(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex = False)
        .str.replace(",", "", regex = False)
        .str.strip(),
        errors = "coerce",
    )


def bucket_review_score(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value < 4.0:
        return "<4.0"
    if value < 4.5:
        return "4.0-4.49"
    if value < 4.7:
        return "4.5-4.69"
    if value < 4.8:
        return "4.7-4.79"
    if value < 4.9:
        return "4.8-4.89"
    if value < 4.95:
        return "4.90-4.94"
    if value < 5.0:
        return "4.95-4.99"
    return "5.0"


def parse_percent(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    values = pd.to_numeric(
        text.str.replace("%", "", regex = False), errors = "coerce"
    ).astype(float)
    percent_format = text.str.contains("%", regex = False, na = False)
    values.loc[percent_format] = values.loc[percent_format] / 100
    values.loc[~percent_format & values.gt(1)] = values.loc[~percent_format & values.gt(1)] / 100
    return values


def parse_amenity_set(value) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        parsed = value
    elif pd.isna(value):
        return set()
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError, TypeError):
            return set()
    if not isinstance(parsed, (list, tuple, set)):
        return set()
    return {str(item).strip().lower() for item in parsed if pd.notna(item)}


def group_property_type(value) -> str:
    text = str(value).lower()
    if any(token in text for token in ["hotel", "hostel", "ryokan", "resort"]):
        return "hotel_or_hostel"
    if any(token in text for token in ["rental unit", "apartment", "condo", "loft"]):
        return "apartment"
    if any(token in text for token in ["home", "house", "townhouse", "villa", "cottage"]):
        return "house"
    if any(token in text for token in ["guesthouse", "bed and breakfast"]):
        return "guesthouse"
    return "other"


def clean_listings(listings: pd.DataFrame) -> pd.DataFrame:
    data = listings.copy()
    data["price_numeric"] = parse_price(data["price"])
    data = data.loc[data["price_numeric"].gt(0)].copy()
    data["log_price"] = np.log(data["price_numeric"].astype(float))

    for column in SCORE_COLUMNS:
        data[f"{column}_bucket"] = data[column].map(bucket_review_score)

    bathroom_text = data["bathrooms_text"].fillna("").str.lower()
    data["bathroom_privacy"] = np.select(
        [bathroom_text.str.contains("shared"), bathroom_text.str.contains("private")],
        ["shared", "private"],
        default = "not_stated",
    )
    data["bathroom_half_bath"] = np.where(
        bathroom_text.str.contains("half-bath"), "yes", "no"
    )
    data["property_type_grouped"] = data["property_type"].map(group_property_type)
    data["minimum_nights_bucket"] = pd.cut(
        data["minimum_nights"],
        bins = [0, 1, 2, 3, 7, 14, 30, np.inf],
        labels = ["1", "2", "3", "4-7", "8-14", "15-30", ">30"],
        include_lowest = True,
    ).astype("string")
    data["host_response_rate_numeric"] = parse_percent(data["host_response_rate"])
    data["host_acceptance_rate_numeric"] = parse_percent(data["host_acceptance_rate"])

    amenity_sets = data["amenities"].map(parse_amenity_set)
    data["amenities_count"] = amenity_sets.map(len)
    for feature, keywords in AMENITY_KEYWORDS.items():
        data[feature] = amenity_sets.map(
            lambda items: "yes"
            if any(keyword in item for item in items for keyword in keywords)
            else "no"
        )

    accommodates = pd.to_numeric(data["accommodates"], errors = "coerce").replace(0, np.nan)
    data["bathrooms_per_guest"] = pd.to_numeric(
        data["bathrooms"], errors = "coerce"
    ) / accommodates
    return data


def aggregate_review_text(
    reviews: pd.DataFrame,
    max_reviews_per_listing: int = 10,
    max_characters_per_review: int = 600,
) -> tuple[pd.Series, pd.Series]:
    data = reviews.dropna(subset = ["listing_id", "comments"]).copy()
    data["date"] = pd.to_datetime(data["date"], errors = "coerce")
    data["comments_clean"] = (
        data["comments"]
        .astype(str)
        .str.replace(r"<br\s*/?>", " ", regex = True, case = False)
        .str.replace(r"<[^>]+>", " ", regex = True)
        .str.replace(r"\s+", " ", regex = True)
        .str.strip()
        .str.slice(0, max_characters_per_review)
    )
    data = data.loc[data["comments_clean"].str.len().gt(0)].sort_values(
        ["listing_id", "date"], ascending = [True, False]
    )
    recent = data.groupby("listing_id", group_keys = False).head(max_reviews_per_listing)
    text = recent.groupby("listing_id")["comments_clean"].agg(" ".join)
    count = recent.groupby("listing_id").size()
    return text, count


def sanitize_text(value) -> str:
    text = str(value).encode("utf-8", errors = "replace").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "missing text"


def add_text_features(
    data: pd.DataFrame,
    review_text: pd.Series,
    review_count: pd.Series,
) -> pd.DataFrame:
    result = data.copy()
    result["listing_text"] = (
        "title " + result["name"].fillna("").astype(str)
        + " description " + result["description"].fillna("").astype(str)
        + " neighborhood " + result["neighborhood_overview"].fillna("").astype(str)
        + " host " + result["host_about"].fillna("").astype(str)
    ).map(sanitize_text)
    result["reviews_text"] = result["id"].map(review_text).fillna("missing text").map(sanitize_text)
    result["amenities_text"] = (
        result["amenities"]
        .map(lambda value: " ".join(sorted(parse_amenity_set(value))) or "missing text")
        .map(sanitize_text)
    )
    result["review_text_count"] = result["id"].map(review_count).fillna(0).astype(float)
    result["listing_text_chars"] = result["listing_text"].str.len().astype(float)
    result["reviews_text_chars"] = result["reviews_text"].str.len().astype(float)
    return result


def prepare_catboost_data(data: pd.DataFrame, include_text: bool = False) -> pd.DataFrame:
    columns = CATBOOST_CATEGORICAL_COLUMNS + CATBOOST_NUMERIC_COLUMNS
    if include_text:
        columns = columns + TEXT_COLUMNS + TEXT_NUMERIC_COLUMNS
    result = data[columns].copy()
    for column in CATBOOST_CATEGORICAL_COLUMNS:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    for column in CATBOOST_NUMERIC_COLUMNS + (TEXT_NUMERIC_COLUMNS if include_text else []):
        result[column] = pd.to_numeric(result[column], errors = "coerce")
    numeric_columns = CATBOOST_NUMERIC_COLUMNS + (TEXT_NUMERIC_COLUMNS if include_text else [])
    result[numeric_columns] = result[numeric_columns].replace([np.inf, -np.inf], np.nan)
    for column in TEXT_COLUMNS if include_text else []:
        result[column] = result[column].fillna("missing text").astype(str)
    return result


def build_design_matrices(
    data: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
) -> dict:
    columns = (
        LINEAR_CATEGORICAL_COLUMNS
        + LINEAR_LOG_NUMERIC_COLUMNS
        + LINEAR_PLAIN_NUMERIC_COLUMNS
    )
    features = data[columns].copy()
    for column in LINEAR_CATEGORICAL_COLUMNS:
        features[column] = features[column].astype("string").fillna("__MISSING__")

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(drop = "first", handle_unknown = "ignore", sparse_output = True)),
    ])
    log_numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out = "one-to-one")),
        ("scale", StandardScaler(with_mean = False)),
    ])
    plain_numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),
        ("scale", StandardScaler(with_mean = False)),
    ])
    spline_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),
        ("spline", SplineTransformer(n_knots = 8, degree = 3, include_bias = False)),
        ("scale", StandardScaler(with_mean = False)),
    ])
    preprocessor = ColumnTransformer(
        transformers = [
            ("categorical", categorical_pipeline, LINEAR_CATEGORICAL_COLUMNS),
            ("log_numeric", log_numeric_pipeline, LINEAR_LOG_NUMERIC_COLUMNS),
            ("plain_numeric", plain_numeric_pipeline, LINEAR_PLAIN_NUMERIC_COLUMNS),
            ("latitude_spline", spline_pipeline, ["latitude"]),
            ("longitude_spline", spline_pipeline, ["longitude"]),
        ],
        sparse_threshold = 1.0,
    )
    train_base = sparse.csr_matrix(preprocessor.fit_transform(features.loc[train_index]))
    test_base = sparse.csr_matrix(preprocessor.transform(features.loc[test_index]))
    base_names = list(preprocessor.get_feature_names_out())

    def dummy_indices(column):
        prefix = f"categorical__{column}_"
        longer = [
            other
            for other in LINEAR_CATEGORICAL_COLUMNS
            if other != column and other.startswith(f"{column}_")
        ]
        excluded = [f"categorical__{other}_" for other in longer]
        return [
            index
            for index, name in enumerate(base_names)
            if name.startswith(prefix) and not any(name.startswith(item) for item in excluded)
        ]

    def feature_index(transformer, column):
        return base_names.index(f"{transformer}__{column}")

    train_interactions = []
    test_interactions = []
    interaction_names = []
    interaction_groups = {}

    def add_interaction(left_indices, right_indices, group_name):
        start = len(base_names) + len(train_interactions)
        for left_index in left_indices:
            for right_index in right_indices:
                train_interactions.append(
                    train_base[:, left_index].multiply(train_base[:, right_index])
                )
                test_interactions.append(
                    test_base[:, left_index].multiply(test_base[:, right_index])
                )
                interaction_names.append(f"{base_names[left_index]} x {base_names[right_index]}")
        interaction_groups[group_name] = np.arange(
            start, start + len(left_indices) * len(right_indices), dtype = int
        )

    add_interaction(
        dummy_indices("neighbourhood_cleansed"),
        dummy_indices("room_type"),
        "Interaction: neighbourhood x room_type",
    )
    add_interaction(
        dummy_indices("neighbourhood_cleansed"),
        dummy_indices("property_type_grouped"),
        "Interaction: neighbourhood x property group",
    )
    add_interaction(
        dummy_indices("room_type"),
        dummy_indices("accommodates"),
        "Interaction: room_type x accommodates",
    )
    add_interaction(
        dummy_indices("property_type_grouped"),
        dummy_indices("accommodates"),
        "Interaction: property group x accommodates",
    )
    add_interaction(
        dummy_indices("bathroom_privacy"),
        dummy_indices("room_type"),
        "Interaction: bathroom privacy x room_type",
    )
    add_interaction(
        [feature_index("log_numeric", "minimum_nights")],
        dummy_indices("room_type"),
        "Interaction: minimum nights x room_type",
    )
    add_interaction(
        [feature_index("log_numeric", "number_of_reviews")],
        dummy_indices("host_is_superhost"),
        "Interaction: reviews x superhost",
    )
    amenity_indices = [index for column in AMENITY_COLUMNS for index in dummy_indices(column)]
    add_interaction(
        amenity_indices,
        dummy_indices("property_type_grouped"),
        "Interaction: amenities x property group",
    )
    for score_column in REVIEW_BUCKET_COLUMNS:
        for count_column in REVIEW_COUNT_COLUMNS:
            add_interaction(
                dummy_indices(score_column),
                [feature_index("log_numeric", count_column)],
                f"Interaction: {score_column} x {count_column}",
            )

    train_matrix = sparse.hstack([train_base] + train_interactions, format = "csr")
    test_matrix = sparse.hstack([test_base] + test_interactions, format = "csr")
    feature_names = base_names + interaction_names

    groups = {}
    for column in LINEAR_CATEGORICAL_COLUMNS:
        indices = np.asarray(dummy_indices(column), dtype = int)
        if len(indices):
            groups[f"Dummy group: {column}"] = indices
    for column in LINEAR_LOG_NUMERIC_COLUMNS:
        groups[f"Numeric: {column}"] = np.asarray(
            [feature_index("log_numeric", column)], dtype = int
        )
    for column in LINEAR_PLAIN_NUMERIC_COLUMNS:
        groups[f"Numeric: {column}"] = np.asarray(
            [feature_index("plain_numeric", column)], dtype = int
        )
    for transformer in ["latitude_spline", "longitude_spline"]:
        indices = np.asarray(
            [index for index, name in enumerate(base_names) if name.startswith(f"{transformer}__")],
            dtype = int,
        )
        groups[f"Spline group: {transformer}"] = indices
    groups.update(interaction_groups)

    return {
        "train": train_matrix,
        "test": test_matrix,
        "feature_names": feature_names,
        "groups": groups,
        "preprocessor": preprocessor,
        "base_feature_count": len(base_names),
        "interaction_count": len(interaction_names),
    }


def fit_groupwise_boosting(
    X_fit,
    y_fit,
    groups: dict,
    max_iter: int,
    learning_rate: float = 0.10,
    X_valid = None,
    y_valid = None,
    patience: int | None = None,
    min_delta: float = 1e-5,
    device: torch.device | None = None,
) -> dict:
    has_validation = X_valid is not None and y_valid is not None
    X_fit_tensor = torch.as_tensor(
        sparse.csr_matrix(X_fit).toarray(), dtype = torch.float32, device = device
    )
    y_fit_tensor = torch.as_tensor(np.asarray(y_fit), dtype = torch.float32, device = device)
    X_valid_tensor = None
    y_valid_tensor = None
    if has_validation:
        X_valid_tensor = torch.as_tensor(
            sparse.csr_matrix(X_valid).toarray(), dtype = torch.float32, device = device
        )
        y_valid_tensor = torch.as_tensor(
            np.asarray(y_valid), dtype = torch.float32, device = device
        )

    group_items = list(groups.items())
    group_names = [name for name, _ in group_items]
    group_indices = [
        torch.as_tensor(indices, dtype = torch.long, device = device)
        for _, indices in group_items
    ]
    n_features = X_fit_tensor.shape[1]
    intercept = y_fit_tensor.mean()
    feature_means = X_fit_tensor.mean(dim = 0)
    X_fit_tensor = X_fit_tensor - feature_means
    if has_validation:
        X_valid_tensor = X_valid_tensor - feature_means

    coefficients = torch.zeros(n_features, dtype = torch.float32, device = device)
    prediction_fit = torch.full_like(y_fit_tensor, float(intercept.item()))
    prediction_valid = (
        torch.full_like(y_valid_tensor, float(intercept.item())) if has_validation else None
    )
    block_solver = torch.zeros((n_features, n_features), dtype = torch.float32, device = device)
    membership = torch.zeros((len(group_items), n_features), dtype = torch.float32, device = device)
    for group_number, indices in enumerate(group_indices):
        X_group = X_fit_tensor[:, indices]
        inverse = torch.linalg.pinv(X_group.T @ X_group, rtol = 1e-5)
        block_solver[indices[:, None], indices[None, :]] = inverse
        membership[group_number, indices] = 1.0

    train_loss = []
    validation_loss = []
    selected_groups = []
    best_iteration = 0
    best_coefficients = coefficients.clone()
    best_validation_loss = (
        torch.mean((y_valid_tensor - prediction_valid) ** 2).item()
        if has_validation
        else np.inf
    )
    iterations_without_improvement = 0

    with torch.no_grad():
        for iteration in range(1, max_iter + 1):
            residual = y_fit_tensor - prediction_fit
            right_hand_side = X_fit_tensor.T @ residual
            group_slopes = block_solver @ right_hand_side
            group_scores = membership @ (right_hand_side * group_slopes)
            selected_number = int(torch.argmax(group_scores).item())
            indices = group_indices[selected_number]
            update = learning_rate * group_slopes[indices]
            coefficients[indices] += update
            prediction_fit += X_fit_tensor[:, indices] @ update
            selected_groups.append(group_names[selected_number])
            train_loss.append(torch.mean((y_fit_tensor - prediction_fit) ** 2).item())

            if has_validation:
                prediction_valid += X_valid_tensor[:, indices] @ update
                current_loss = torch.mean((y_valid_tensor - prediction_valid) ** 2).item()
                validation_loss.append(current_loss)
                if current_loss < best_validation_loss - min_delta:
                    best_validation_loss = current_loss
                    best_iteration = iteration
                    best_coefficients = coefficients.clone()
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
                if patience is not None and iterations_without_improvement >= patience:
                    break
            else:
                best_iteration = iteration
                best_coefficients = coefficients.clone()

    result = {
        "intercept": float(intercept.cpu()),
        "feature_means": feature_means.cpu().numpy(),
        "coefficients": best_coefficients.cpu().numpy(),
        "best_iteration": best_iteration,
        "iterations_run": len(train_loss),
        "train_loss": np.asarray(train_loss),
        "validation_loss": np.asarray(validation_loss),
        "selected_groups": selected_groups,
    }
    del X_fit_tensor, X_valid_tensor, block_solver, membership
    torch.cuda.empty_cache()
    return result


def predict_groupwise_boosting(model: dict, X_new) -> np.ndarray:
    matrix = sparse.csr_matrix(X_new, dtype = float)
    linear_part = np.asarray(matrix @ model["coefficients"]).ravel()
    offset = model["feature_means"] @ model["coefficients"]
    return model["intercept"] + linear_part - offset


def evaluate_predictions(model_name: str, y_true_log, y_pred_log) -> dict:
    y_true_log = np.asarray(y_true_log)
    y_pred_log = np.asarray(y_pred_log)
    y_true_price = np.exp(y_true_log)
    y_pred_price = np.exp(y_pred_log)
    return {
        "Model": model_name,
        "MSE log(price)": mean_squared_error(y_true_log, y_pred_log),
        "MAE log(price)": mean_absolute_error(y_true_log, y_pred_log),
        "R2 log(price)": r2_score(y_true_log, y_pred_log),
        "MAE price": mean_absolute_error(y_true_price, y_pred_price),
        "Median AE price": median_absolute_error(y_true_price, y_pred_price),
    }


CATBOOST_GPU_PARAMETERS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 4_000,
    "learning_rate": 0.10,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "boosting_type": "Plain",
    "border_count": 32,
    "leaf_estimation_iterations": 1,
    "max_ctr_complexity": 1,
    "one_hot_max_size": 10,
    "random_seed": RANDOM_STATE,
    "task_type": "GPU",
    "devices": "0",
    "gpu_ram_part": 0.85,
    "allow_writing_files": False,
}


def fit_catboost(
    model_name: str,
    fit_pool: Pool,
    validation_pool: Pool,
    early_stopping_rounds: int = 200,
    verbose: int = 200,
    **extra_parameters,
) -> tuple[CatBoostRegressor, float]:
    parameters = {**CATBOOST_GPU_PARAMETERS, **extra_parameters}
    model = CatBoostRegressor(**parameters)
    start = perf_counter()
    model.fit(
        fit_pool,
        eval_set = validation_pool,
        use_best_model = True,
        early_stopping_rounds = early_stopping_rounds,
        verbose = verbose,
    )
    seconds = perf_counter() - start
    validation_mse = model.get_best_score()["validation"]["RMSE"] ** 2
    print(f"{model_name}: {seconds / 60:.2f} minutes")
    print(f"Best iteration: {model.get_best_iteration() + 1}")
    print(f"Validation MSE: {validation_mse:.6f}")
    return model, seconds


def build_tfidf_features(
    text_train: pd.DataFrame,
    text_test: pd.DataFrame,
    fit_positions: np.ndarray,
    validation_positions: np.ndarray,
) -> dict:
    def combine(frame):
        return (
            "listing " + frame["listing_text"].astype(str)
            + " reviews " + frame["reviews_text"].astype(str)
            + " amenities " + frame["amenities_text"].astype(str)
        )

    train_documents = combine(text_train)
    test_documents = combine(text_test)
    fit_documents = train_documents.iloc[fit_positions]
    validation_documents = train_documents.iloc[validation_positions]

    word_vectorizer = TfidfVectorizer(
        analyzer = "word",
        ngram_range = (1, 2),
        min_df = 5,
        max_df = 0.98,
        max_features = 12_000,
        sublinear_tf = True,
        dtype = np.float32,
    )
    character_vectorizer = TfidfVectorizer(
        analyzer = "char_wb",
        ngram_range = (3, 5),
        min_df = 5,
        max_features = 12_000,
        sublinear_tf = True,
        dtype = np.float32,
    )
    word_fit = word_vectorizer.fit_transform(fit_documents)
    character_fit = character_vectorizer.fit_transform(fit_documents)
    fit_sparse = sparse.hstack([word_fit, character_fit], format = "csr")

    def transform(documents):
        return sparse.hstack(
            [word_vectorizer.transform(documents), character_vectorizer.transform(documents)],
            format = "csr",
        )

    validation_sparse = transform(validation_documents)
    test_sparse = transform(test_documents)
    n_components = min(128, fit_sparse.shape[0] - 1, fit_sparse.shape[1] - 1)
    svd = TruncatedSVD(n_components = n_components, n_iter = 5, random_state = RANDOM_STATE)
    fit_svd = svd.fit_transform(fit_sparse).astype(np.float32)
    validation_svd = svd.transform(validation_sparse).astype(np.float32)
    test_svd = svd.transform(test_sparse).astype(np.float32)
    columns = [f"tfidf_svd_{number:03d}" for number in range(n_components)]
    return {
        "fit": pd.DataFrame(fit_svd, index = text_train.iloc[fit_positions].index, columns = columns),
        "validation": pd.DataFrame(
            validation_svd, index = text_train.iloc[validation_positions].index, columns = columns
        ),
        "test": pd.DataFrame(test_svd, index = text_test.index, columns = columns),
        "columns": columns,
        "word_vocabulary": len(word_vectorizer.vocabulary_),
        "character_vocabulary": len(character_vectorizer.vocabulary_),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
    }


def build_clip_prompt_features(
    data: pd.DataFrame,
    manifest_path: Path,
    embedding_path: Path,
    device: torch.device,
    prompts: list[str] = IMAGE_PROMPTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import clip

    manifest = pd.read_csv(manifest_path, low_memory = False)
    with np.load(embedding_path, allow_pickle = False) as cache:
        cached_paths = cache["local_paths"].astype(str)
        cached_vectors = cache["embeddings"].astype(np.float32)
    path_to_row = {path: row for row, path in enumerate(cached_paths)}
    listing_to_path = manifest.drop_duplicates("listing_id").set_index("listing_id")["local_path"]
    listing_paths = data["id"].map(listing_to_path)
    image_matrix = np.zeros((len(data), cached_vectors.shape[1]), dtype = np.float32)
    available = np.zeros(len(data), dtype = np.float32)
    for row_number, path in enumerate(listing_paths):
        cached_row = path_to_row.get(str(path))
        if cached_row is not None:
            image_matrix[row_number] = cached_vectors[cached_row]
            available[row_number] = 1.0

    model, _ = clip.load("ViT-B/32", device = device, jit = False)
    model.eval()
    tokens = clip.tokenize(prompts).to(device)
    with torch.inference_mode(), torch.autocast(device_type = "cuda", dtype = torch.float16):
        text_embeddings = model.encode_text(tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(
            dim = 1, keepdim = True
        ).clamp_min(1e-12)
    scores = image_matrix @ text_embeddings.float().cpu().numpy().T
    scores[available == 0] = 0.0
    del model, tokens, text_embeddings
    torch.cuda.empty_cache()

    columns = [f"clip_prompt_{number:02d}" for number in range(len(prompts))]
    features = pd.DataFrame(scores, index = data.index, columns = columns)
    features["clip_image_available"] = available
    prompt_map = pd.DataFrame({"Feature": columns, "Prompt": prompts})
    return features, prompt_map
