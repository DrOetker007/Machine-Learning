# Airbnb Price Prediction Project
Welcome to the Airbnb Price Prediction Project—the final project for the SoSe 2026 Machine Learning course. Over the next six weeks you’ll replace the usual three exercise sheets with one bigger project: predicting nightly listing prices. Plan on dedicating roughly one full working day each week (~6 days total). This guide summarizes the tasks; work in whichever environment suits you best.

**Dataset at a glance.**
Your raw material comes from the open-data initiative Inside Airbnb, which regularly scrapes the public Airbnb site and publishes cleansed snapshots of listings, reviews, calendars, and photos for dozens of cities. It’s a well-documented, multimodal dataset that’s refreshed several times a year.

**Why it matters.**
This project provides an opportunity to develop your first machine learning project.: beginning with real-world data acquisition and exploration, deciding how to wrangle and enrich that data, selecting and comparing methods, and iterating. The process reassembles how many industry-scale ML problems evolve—from raw data to actionable models.
### Objective

Your task is to build a predictive model that estimates listing prices for Airbnb accommodations. You will:

1. **Select a City**: Choose one city from the datasets available on the [Inside Airbnb website](https://insideairbnb.com/get-the-data/).
2. **Download Data**: Obtain the relevant files (listings, reviews, calendar, and images) for your chosen city.
3. **Incorporate Multiple Modalities**: Use at least two data forms, see the suggestions below:
   - **Tabular Data**: Structured listing and calendar data
   - **Text Data**: Descriptions, reviews, and other textual fields
   - **Image Data**: Photos of the listings (Photos are referenced by URL)
   - **Spatial Data**: Coordinates of the listings
4. **Build a Predictive Model**: Compare and combine different modeling approaches. Examples include, but are not limited to:
   - Linear or polynomial regression
   - Random forests or gradient-boosted trees
   - Neural networks (e.g., multi-layer perceptron, convolutional networks, transformers)
   - Hybrid models (e.g., text embeddings from a language model combined with a regression or tree-based model)
5. **Evaluation**: Split your dataset into training and testing subsets. Evaluate models using appropriate metrics and compare their performance.

### Workflow Suggestions

1. **Data Cleaning & Preparation**
   - Handle missing or inconsistent values.
   - Create derived features (e.g. review sentiment scores).
   - Normalize or scale numerical variables.
2. **Feature Engineering**
   - Encode categorical variables (e.g., one-hot encoding for neighborhood).
   - Extract text features (e.g., TF-IDF, pretrained embeddings,...).
   - Process images (e.g., resize, extract embeddings,...).
3. **Modeling & Tuning**
   - Model creation
   - Experiment with hyperparameter tuning.
   - Evaluate models on a held-out dataset to assess generalization.
4. **Analysis & Reporting**
   - Document your findings with clear visualizations (plots, tables).
   - Discuss trade-offs between models, modalities, and feature choices.
   - Suggest possible improvements.

### Deliverables
- **Intermediate Presentation (Jun 24–26)**
   -  Is the option for you to present preliminary ideas and results; ask for feedback; tool use...
   -  Talk about problems already encountered
   -  Pitch your next steps: How do you want to proceed with the project? What kind of methods do you want to apply...
- **Final Presentation (Jul 15-17)**:
   - How will it work:
     - In the normal time slot, each group will present their final project to one of the tutors.
     - Prepare a slide deck to briefly guide your tutor through your project (5–10 min presentation, followed by open Q&A with your tutor)
   - Content example guideline but feel free to deviate
     - Project objective (which city is chosen)
     - Preprocessing steps
     - Model descriptions and evaluation results
     - Discussion of results and conclusions
- **Code**: A well organized code repository (not everything in a single jupyter notebook, separated by task). Upload the reposetory to the project folder in StudIP until Jul 15.


**Remark:** There is no single "correct" approach. Focus on a robust, reproducible workflow, thoughtful feature engineering, and clear communication of your results. Good luck!
