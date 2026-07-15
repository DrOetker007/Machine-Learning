# Airbnb Price Prediction in Tokyo

This project predicts the nightly asking price of Tokyo Airbnb listings from an Inside Airbnb snapshot dated September 29/30, 2025. It combines tabular, spatial, text, and main-image features and compares linear regression, component-wise boosting, histogram gradient boosting, and several CatBoost variants.

## Project structure

| Path | Purpose |
| --- | --- |
| `airbnb_price_prediction.ipynb` | Main analysis and final report. |
| `airbnb_project.py` | Reusable cleaning, feature-engineering, modeling, and evaluation helpers. |
| `download_listing_images.py` | Resumable download of listing main images and creation of the image manifest. |
| `extract_clip_embeddings.py` | Extraction and caching of normalized CLIP image embeddings. |
| `data/` | Inside Airbnb listings, reviews, calendar, and spatial data. |
| `presentation/` | LaTeX source, figures, and PDF of the final presentation. |
| `requirements.txt` | Python dependencies used for the analysis. |

The downloaded images, `data/image_manifest.csv`, and `data/clip_main_image_embeddings.npz` are not included because of their size. They can be rebuilt with the preparation scripts below.

## Setup

Run all commands from the `Final_Project` directory. The analysis was run with Python 3.10 on an NVIDIA GPU with CUDA 12.1. Git is required to install CLIP from its repository, and the main notebook and CLIP extraction require CUDA.

The compressed reviews file is stored with Git LFS. After cloning the repository, make sure that the data file has been downloaded:

```powershell
git lfs install
git lfs pull
```

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The CLIP dependency is installed directly from a fixed GitHub revision. Its model weights are downloaded automatically on first use.

## Reproduce the analysis

The raw CSV files are already stored in `data/`. Rebuild the ignored image artifacts and then open the main notebook:

```powershell
python download_listing_images.py
python extract_clip_embeddings.py
jupyter lab airbnb_price_prediction.ipynb
```

The image downloader is resumable. Because the source images are hosted externally, individual old URLs may no longer be available; missing images are represented explicitly by the image-availability feature.

The main notebook uses a fixed random seed and an 80/20 listing-level train-test split. Early stopping uses a separate validation subset of the training data. All reported model metrics are calculated on the held-out test set. The evaluation therefore applies to unseen listings from the same Tokyo market; hosts may occur in both training and test data.
