# 📉 Telecom Customer Churn Prediction

---

## 🔍 Project Overview
Customer Churn (customer attrition) is one of the biggest challenges in the telecommunications industry. Retaining existing customers is significantly more cost-effective than acquiring new ones. 

This project utilizes **Machine Learning (Random Forest Classifier)** to predict which customers are likely to cancel their service. By analyzing demographics, account information, and service usage, we generate insights that can help the company take proactive retention measures.

---

## 🏢 Business Context
The goal of this project is not just to predict churn, but to understand the **factors driving it**. 
* **Problem:** High churn rates reduce revenue and profitability.
* **Solution:** A predictive model that identifies high-risk customers.
* **Impact:** Allows the marketing team to target specific customers with retention offers (discounts, contract upgrades) before they leave.

---

## 📂 Dataset Description
The dataset used is the standard Telco Customer Churn dataset. It contains **7,043 rows** (customers) and **21 features**.

| Feature Type | Examples |
| :--- | :--- |
| **Target** | `Churn` (Yes = Left, No = Stayed) |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Services** | `PhoneService`, `InternetService`, `OnlineSecurity`, `StreamingTV` |
| **Account Info** | `Tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Financials** | `MonthlyCharges`, `TotalCharges` |

---

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Random Forest, RandomizedSearchCV)

---

## ⚙️ Project Workflow

### 1. Data Cleaning
* **Handling Data Types:** The `TotalCharges` column contained blank strings representing missing values. These were coerced to `NaN` and rows with missing values were dropped (accounting for <0.2% of data).
* **Leakage Prevention:** The `customerID` column was immediately removed as it provides no predictive value and causes model overfitting/data leakage.

### 2. Exploratory Data Analysis (EDA)
* Analyzed the class imbalance: ~27% of customers churned vs. 73% retained.
* Identified that month-to-month contracts have the highest churn rate.

### 3. Feature Engineering & Preprocessing
* **Target Encoding:** Mapped `Churn` to binary (1/0).
* **One-Hot Encoding:** Applied `pd.get_dummies` to categorical variables (e.g., `PaymentMethod`, `InternetService`) with `drop_first=True` to avoid multicollinearity.
* **Feature Selection:** All preprocessed features were utilized.

### 4. Model Training
* **Algorithm:** Random Forest Classifier.
* **Strategy:** Used `class_weight='balanced'` to penalize misclassification of the minority class (Churners).
* **Hyperparameter Tuning:** Utilized **`RandomizedSearchCV`** to optimize:
    * `n_estimators` (Number of trees)
    * `max_depth` (Tree depth to control overfitting)
    * `min_samples_split` & `min_samples_leaf`
    * `criterion` (Gini vs. Entropy)

---

## 📊 Model Performance
The model was evaluated on a 30% hold-out test set.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **~79%** | Overall correctness of the model. |
| **ROC-AUC** | **0.84** | Excellent ability to distinguish between Churners and Non-Churners. |
| **Recall (Churn)** | **~0.75** | The model correctly identifies 75% of customers who actually left. |

*Note: High recall is prioritized in churn prediction because missing a churning customer (False Negative) is more expensive than offering a discount to a loyal customer (False Positive).*

---

## 💡 Key Business Insights
Based on **Feature Importance** analysis, the top drivers of churn are:

1.  **Contract Type:** Customers on **Month-to-Month** contracts are the highest risk. Long-term contracts (1-2 years) significantly reduce churn.
2.  **Tenure:** New customers (low tenure) are volatile. Loyalty increases drastically after the first year.
3.  **Total Charges:** Customers with higher lifetime spend are actually *more* likely to churn, possibly due to price sensitivity or finding better deals elsewhere.
4.  **Internet Service:** Users with **Fiber Optic** service have higher churn rates than DSL users, suggesting potential dissatisfaction with service quality or price.

---

## 🚀 Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Shashank-2207/telecom-churn-prediction.git](https://github.com/Shashank-2207/telecom-churn-prediction.git)
    ```

2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
    ```

3.  **Run the Analysis:**
    Open the Jupyter Notebook or run the Python script:
    ```bash
    jupyter notebook Random_Forest_model.ipynb
    ```

---

## 🔮 Future Improvements
* **Model Comparison:** Test XGBoost, LightGBM, and Logistic Regression to see if performance improves.
* **SMOTE:** Use Synthetic Minority Over-sampling Technique to further address class imbalance.
* **Deployment:** Create a simple Flask or Streamlit API to allow end-users to input customer data and get a churn probability score.

---
