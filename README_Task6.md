# Task 6 — House Price Prediction

**DevelopersHub Corporation — AI/ML Engineering Internship**  
**Author:** Abdul Samad  
**Date:** May 2026

---

## Objective

Build a regression pipeline to predict residential house prices using structural and amenity features. Train and compare two models — Linear Regression and Gradient Boosting — and evaluate performance using MAE, RMSE, and R².

---

## Dataset

| Detail | Value |
|---|---|
| Source | [Housing Prices Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) |
| File | Housing.csv |
| Rows | 545 |
| Features | 12 |
| Target | `price` |
| Currency | Indian Rupees (INR / ₹) |
| Price Range | ₹17,50,000 — ₹1,33,00,000 |
| Missing Values | None |

---

## Features Used

| Feature | Type | Description |
|---|---|---|
| area | Numerical | Total area in square feet |
| bedrooms | Numerical | Number of bedrooms |
| bathrooms | Numerical | Number of bathrooms |
| stories | Numerical | Number of floors |
| parking | Numerical | Number of parking spaces |
| mainroad | Binary (1/0) | Connected to main road |
| guestroom | Binary (1/0) | Has a guest room |
| basement | Binary (1/0) | Has a basement |
| hotwaterheating | Binary (1/0) | Has hot water heating |
| airconditioning | Binary (1/0) | Has air conditioning |
| prefarea | Binary (1/0) | Located in a preferred area |
| furnishingstatus | Encoded (0/1/2) | unfurnished / semi-furnished / furnished |

---

## Libraries and Tools

- Python 3.11
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

---

## Steps Performed

1. Loaded and inspected the dataset (shape, types, missing values)
2. Encoded binary yes/no columns to 1/0
3. Encoded `furnishingstatus` with ordered mapping (unfurnished=0, semi-furnished=1, furnished=2)
4. Exploratory data analysis — price distribution, price vs area scatter, correlation heatmap
5. 80/20 train-test split (436 train, 109 test)
6. Applied StandardScaler — fit on train only, applied to test
7. Trained Linear Regression and Gradient Boosting
8. Evaluated both models on the held-out test set
9. Plotted actual vs predicted and feature importances
10. Predicted price for a sample house

---

## Results

All prices in Indian Rupees (INR).

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ₹9,79,680 | ₹13,31,071 | 0.6495 |
| **Gradient Boosting** | **₹9,60,220** | **₹13,10,960** | **0.6600** |

**Winner: Gradient Boosting**

Gradient Boosting outperformed Linear Regression across all three metrics. The margin is small, but consistent — confirming that house price relationships have some non-linearity that a linear model cannot capture.

---

## Key Findings

**1. Area is the dominant predictor**  
Both models agree that `area` (square footage) carries the most predictive weight. Larger houses command higher prices consistently across the dataset.

**2. Air conditioning and preferred area add significant price premium**  
`airconditioning` and `prefarea` ranked second and third in Gradient Boosting feature importance — binary features but high signal.

**3. Gradient Boosting handles non-linearity; Linear Regression does not**  
Unlike the stock price task (Task 2) where AAPL prices moved smoothly and Linear Regression won, house prices are driven by a combination of features interacting non-linearly — which is Gradient Boosting's strength.

**4. R² of 0.66 reflects the dataset's limitations, not model failure**  
The dataset contains no location data beyond a single binary `prefarea` column. In real estate, location alone typically explains 40-60% of price variance. With only yes/no location information, a ceiling around 0.65-0.70 is expected on this dataset.

**5. Hot water heating has near-zero predictive value**  
`hotwaterheating` ranked last in both models — present in very few houses and not correlated with price in this dataset.

---

## Model Configuration

**Linear Regression**  
Default scikit-learn configuration. No hyperparameters required.

**Gradient Boosting**
```
n_estimators  = 300
max_depth     = 4
learning_rate = 0.05
subsample     = 0.8
random_state  = 42
```

---

## How to Run

```bash
# 1. Create and activate virtual environment
python -m venv task6
task6\Scripts\activate          # Windows
source task6/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel

# 3. Register kernel and launch notebook
python -m ipykernel install --user --name=task6 --display-name "Task6"
jupyter notebook

# 4. Open house_price_prediction.ipynb and run all cells
```

Place `Housing.csv` in the same directory as the notebook before running.

---

## Repository Structure

```
task6-house-price-prediction/
│
├── house_price_prediction.ipynb   # Main notebook
├── Housing.csv                    # Dataset
├── eda_plots.png                  # EDA visualizations
├── actual_vs_predicted.png        # Model comparison plot
├── feature_importance.png         # Feature importance chart
└── README.md                      # This file
```

---

## Limitations

- No granular location data — the single `prefarea` binary column is a weak substitute for actual location/neighbourhood information.
- Small dataset (545 rows) limits Gradient Boosting from reaching its full potential.
- Prices are in INR and specific to an Indian housing context — models trained here would not generalise to other markets.

---

## Author

**Abdul Samad**  
DevelopersHub Corporation — AI/ML Engineering Internship  
GitHub: [mr3ansar](https://github.com/mr3ansar)
