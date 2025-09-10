# Fight-Delay-
# ✈️ Flight Arrival Delay Prediction

## 📌 Overview
This project builds and evaluates machine learning models to predict **flight arrival delays** using real-world flight data.  
The pipeline covers **data preprocessing, feature engineering, model training, and evaluation** with visualizations.  

---

## 📂 Workflow

1. **Data Loading**
   - Load dataset from `flightuu.csv` using pandas.  
   - Inspect dataset (`head()`, `info()`, `isnull().sum()`).

2. **Data Cleaning**
   - Drop irrelevant columns (e.g., flight date, cancellation codes, etc.).  
   - Handle missing values by removing rows with NaNs.  
   - Apply **one-hot encoding** to categorical features (`AIRLINE`, `ORIGIN`, `DEST`).  

3. **Feature & Target Definition**
   - Features = all columns except `ARR_DELAY`.  
   - Target = `ARR_DELAY` (arrival delay in minutes).  

4. **Train-Test Split**
   - 80% training, 20% testing.  

5. **Models Implemented**
   - **Linear Regression**
     - Evaluate with R² and Mean Squared Error (MSE).  
     - Plot: Actual vs Predicted, Residuals.  
   - **Decision Tree Regressor**
     - Evaluate R² and MSE.  
     - Visualize partial tree structure and predictions.  
   - **Random Forest Regressor**
     - Ensemble approach with 100 trees, max depth = 10.  
     - Evaluate R² and MSE.  

6. **Model Comparison**
   - Simulated cross-validation accuracy scores for:
     - Linear Regression  
     - Decision Tree  
     - KNN Classifier  
     - Random Forest  
   - Boxplot visualization for model performance comparison.  

---

## ⚙️ Requirements

Install the required libraries:

```bash
pip install pandas seaborn scikit-learn matplotlib numpy
```

---

## ▶️ Usage

1. Place your dataset as `flightuu.csv` in the project directory.  
2. Run the script:

```bash
python Flight.py
```

3. The script will:
   - Preprocess the dataset.  
   - Train models (Linear Regression, Decision Tree, Random Forest).  
   - Print evaluation metrics.  
   - Display visualizations (scatter plots, residual plots, decision tree, and comparison boxplot).  

---

## 📊 Expected Outputs

### 🔹 Metrics
Example output in the terminal (values may vary depending on dataset):

```
Linear Regression:
Train R² Score: 0.92
Test R² Score: 0.89
Test Mean Squared Error: 45.32

Decision Tree:
Train R² Score: 1.00
Test R² Score: 0.87
Test Mean Squared Error: 50.11

Random Forest:
Train R² Score: 0.97
Test R² Score: 0.91
Test Mean Squared Error: 42.78
```

### 🔹 Visualizations
- **Linear Regression:**
  - Scatter plot of Actual vs Predicted arrival delays.  
  - Residual plot to check error distribution.  

- **Decision Tree:**
  - Partial decision tree visualization (limited depth for readability).  
  - Actual vs Predicted scatter plot.  

- **Random Forest:**
  - Scatter plot comparing predicted vs actual delays.  

- **Comparison:**
  - Boxplot of simulated accuracy scores for Linear Regression, Decision Tree, KNN, and Random Forest.  

---

## 📌 Notes

- This project demonstrates regression-based approaches to predict flight delays.  
- Feature engineering and hyperparameter tuning can improve accuracy.  
- KNN results are simulated here; you can implement a full KNN model for actual evaluation.  
