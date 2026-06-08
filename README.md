# Personal-Loan-Prediction-Model
This project uses the Bank Personal Loan Modeling dataset to apply predictive analytics and classification modeling. The goal is to identify customers who are most likely to accept a personal loan offer based on their demographic and financial characteristics.

## Key Steps

1. Imported and explored the Bank Personal Loan Modeling dataset.
2. Audited the data for missing values, duplicates, and data quality issues.
3. Removed non-predictive variables such as ID and ZIP Code.
4. Performed data profiling using descriptive statistics and visualizations.
5. Split the data into training and testing datasets.
6. Applied a Decision Tree Classification model.
7. Evaluated model performance using accuracy, classification reports, and confusion matrices.
8. Identified the most important variables influencing loan acceptance.
9. Generated business recommendations based on model findings.

## 📷 Project Screenshots

### Personal Loan Distribution
<img width="560" height="528" alt="image" src="https://github.com/user-attachments/assets/ab1a3d83-9dbd-4ece-ac82-6234b00daf10" />


### Feature Importance Analysis

<img width="771" height="631" alt="image" src="https://github.com/user-attachments/assets/584e8f50-1ade-4da0-b09e-857d195cfc35" />


### Decision Tree Model

<img width="975" height="468" alt="image" src="https://github.com/user-attachments/assets/28fe0e97-7ff9-468e-bf78-3d40e8191346" />

### Confusion Matrix


<img width="595" height="516" alt="image" src="https://github.com/user-attachments/assets/8c740bbf-ccaa-4317-b178-a97ccbb4eabb" />



## Sample Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel(
    "Bank_Personal_Loan_Modelling.xlsx",
    sheet_name="Data"
)

# Remove non-predictive variables
df = df.drop(columns=["ID", "ZIP Code"])

# Define features and target
X = df.drop(columns=["Personal Loan"])
y = df["Personal Loan"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Train model
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:",
      accuracy_score(y_test, y_pred))
```

## Model Findings

The Decision Tree model identified several important predictors of personal loan acceptance:

- Income
- CD Account Ownership
- Average Credit Card Spending (CCAvg)
- Education Level
- Family Size

Customers with higher income levels and stronger relationships with the financial institution were more likely to accept personal loan offers.

## Skills Demonstrated

- Data Cleaning
- Data Auditing
- Exploratory Data Analysis (EDA)
- Feature Selection
- Classification Modeling
- Decision Trees
- Model Evaluation
- Business Analytics
- Financial Services Analytics
- Python

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Google Colab
- Jupyter Notebook

## Business Value

This project demonstrates how predictive analytics can help financial institutions:

- Improve marketing efficiency
- Increase loan conversion rates
- Reduce customer acquisition costs
- Better target high-value customers
- Support data-driven decision making

## Author

**Victoria Riley**

Graduate Student | Data Analytics & Predictive Modeling

GitHub: [@vriley31](https://github.com/vriley31)
