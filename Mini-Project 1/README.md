# Home Credit Default Risk Prediction

This is our Mini-Project 1 for MSDM 5054. We use the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data) to predict whether a loan applicant will have difficulty repaying a loan. We compare Logistic Regression, Random Forest, and XGBoost using five-fold cross-validation.

## Project Structure

```text
Mini-Project 1/
├── data/
│   ├── raw/                     # Original Kaggle data
│   └── processed/               # Cleaned and engineered datasets
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_data_preprocess.ipynb # Data cleaning and preprocessing
│   ├── 03_feature_engineering.ipynb
│   ├── 04a_logistic_regression.ipynb
│   ├── 04b_random_forest.py
│   ├── 04c_xgboost_model.py     # XGBoost model settings
│   ├── 04c_tune_xgboost.py      # Hyperparameter tuning
│   ├── 04c_run_xgboost.py       # Standalone XGBoost run
│   └── 04c_run_xgboost_5cv.py  # Five-fold XGBoost evaluation
├── configs/                     # Saved model configuration
└── results/                     # Saved model evaluation results
    ├── logistic_regression/
    ├── random_forest/
    └── xgboost/
```

## Requirements

- Python 3.14.7
- Required packages are listed in `requirements.txt`.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Data and Code

The raw and processed datasets are not included in this repository because of their size. The original data can be downloaded from the Kaggle link above and placed in `data/raw/`.

The main workflow is:

1. Run `01_eda.ipynb` to explore the dataset.
2. Run `02_data_preprocess.ipynb` to clean the application data.
3. Run `03_feature_engineering.ipynb` to create the final modeling datasets.
4. Use the Logistic Regression, Random Forest, and XGBoost files for model training and evaluation.

The notebooks use paths relative to the `notebooks/` folder. Some Python scripts were run in local environments, so their data paths or supporting modules may need to be adjusted before running them elsewhere. Model evaluation summaries are available in `results/`.

## Contributions

| Name | GitHub | Contribution |
|------|--------|--------------|
| WANG, Yuxun | [@Pigeonee](https://github.com/Pigeonee) | Conducted EDA, data preprocessing, feature engineering, and Logistic Regression modeling; contributed to the Introduction, Dataset and Data Preparation, Methodology, and Logistic Regression results sections of the report. |
| ZHENG, Yuanhan | [@F1ameengo](https://github.com/F1ameengo) | Constructed Random Forest Model; contributed to the Abstract, Methodology, Random Forest results, Conclusion and Discussion section of the report. |
| YU, Ruihan | [@yunk9617-ship-it](https://github.com/yunk9617-ship-it) | Construct XGBoost Model, and refine the parameter. Contribute to the Methodology, Result, Analysis section of the report. |
