# Kaggle Playground Series S5E9: Predicting Beats Per Minute (BPM)

A machine learning project to predict the Beats Per Minute (BPM) of music tracks using various audio and musical features.

## Project Overview

This project tackles the Kaggle Playground Series Season 5, Episode 9 competition, where the goal is to predict the BPM of music tracks based on various audio characteristics and musical features. The solution uses LightGBM with hyperparameter optimization to achieve accurate predictions.

### Data Statistics
- NO duplicates
- No missing values


## Technical Approach

### 1. Exploratory Data Analysis (EDA)
- **Distribution Analysis**: Examined target variable and feature distributions
- **Outlier Detection**: Used box plots and z-score analysis (3σ threshold)
- **Correlation Analysis**: Heatmap visualization to understand feature relationships
- **Feature Quality Assessment**: Verified data integrity and identified patterns

### 2. Feature Engineering
Applied comprehensive feature engineering techniques:
- **Interaction Features**: Created combinations like `MoodScore × RhythmScore`
- **Polynomial Features**: Added squared terms for non-linear relationships
- **Outlier Flags**: Binary indicators for outlier detection
- **Ratio Features**: Energy-to-duration ratios
- **Log Transformations**: Applied to skewed features
- **Temporal Features**: Duration conversions and time-based features

### 3. Model Development
- **Algorithm**: LightGBM Regressor
- **Hyperparameter Optimization**: RandomizedSearchCV with 100 iterations
- **Cross-Validation**: 5-fold CV for robust parameter selection
- **Feature Selection**: Kept all engineered features based on correlation analysis

## Getting Started

### Prerequisites
```bash
pip install polars pandas numpy matplotlib seaborn scikit-learn lightgbm
```

### Running the Project
1. Clone the repository
2. Place the competition data files (`train.csv`, `test.csv`) in the project directory
3. Run the Jupyter notebook `BMP_predict.ipynb`
4. The model will generate `submission_lightgbm.csv` for Kaggle submission

### Project Structure
```
├── BMP_predict.ipynb          # Main analysis and modeling notebook
├── train.csv                  # Training dataset
├── test.csv                   # Test dataset
├── sample_submission.csv      # Sample submission format
├── submission_lightgbm.csv    # Generated predictions
└── README.md                  # Project documentation
```

## Model Pipeline

1. **Data Loading & Preprocessing**
   - Load data using Polars for efficient processing
   - Basic data quality checks

2. **Exploratory Data Analysis**
   - Target variable distribution analysis
   - Feature correlation analysis
   - Outlier detection and visualization

3. **Feature Engineering**
   - Create interaction and polynomial features
   - Apply transformations for better model performance
   - Generate ratio and temporal features

4. **Model Training**
   - Train LightGBM with optimized hyperparameters
   - Use RandomizedSearchCV for parameter tuning
   - Cross-validation for robust evaluation

5. **Model Evaluation**
   - Multiple regression metrics (RMSE, MAE, R²)
   - Feature importance analysis
   - Prediction vs actual visualization

6. **Prediction Generation**
   - Generate predictions for test set
   - Create submission file in required format

## Dependencies

- `polars`: Efficient data manipulation
- `pandas`: Data analysis and manipulation
- `numpy`: Numerical computations
- `matplotlib/seaborn`: Data visualization
- `scikit-learn`: Machine learning utilities
- `lightgbm`: Gradient boosting framework
