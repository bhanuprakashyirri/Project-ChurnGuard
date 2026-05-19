# 🛡️ Project ChurnGuard — Customer Churn Prediction System

ChurnGuard is a machine learning project that predicts whether a telecom customer is likely to **churn** (leave the service) or **stay**, based on their account details and usage patterns. It follows a clean, step-by-step pipeline from raw data exploration to real-time prediction.

---

## 📁 Project Structure

```
Project-ChurnGuard/
│
├── churnguard_data.csv        # Raw customer dataset
├── task1_load_explore.py      # Step 1: Load & explore the dataset
├── task2_clean_data.py        # Step 2: Clean and preprocess the data
├── task3_train_model.py       # Step 3: Train a Logistic Regression model
├── task4_predict.py           # Step 4: Interactive churn prediction
└── .gitignore
```

---

## 🔄 Workflow

### Task 1 — Load & Explore (`task1_load_explore.py`)
- Loads the CSV dataset using pandas
- Prints dataset shape, first 5 rows, column data types
- Reports missing values and duplicate rows
- Shows value counts for the `Churn` column and unique values in `Contract`

### Task 2 — Clean Data (`task2_clean_data.py`)
- Drops unnecessary columns (`customerID`) and duplicate rows
- Standardises categorical values for `Contract`, `InternetService`, `Churn`, `PhoneService`, `PaperlessBilling`, and `PaymentMethod`
- Converts `TotalCharges` to numeric
- Filters out invalid records (e.g., zero or negative `tenure`, out-of-range `MonthlyCharges`)
- Fills missing values using mean (for numeric) and mode (for categorical)

### Task 3 — Train Model (`task3_train_model.py`)
- Applies all cleaning steps from Task 2
- Encodes the target column `Churn` as binary (Yes → 1, No → 0)
- One-hot encodes categorical features using `pd.get_dummies`
- Splits data into 80% train / 20% test sets
- Scales features with `StandardScaler`
- Trains a **Logistic Regression** model (`class_weight='balanced'` to handle class imbalance)
- Evaluates with **accuracy score** and a full **classification report**

### Task 4 — Predict (`task4_predict.py`)
- Retrains the model on the full dataset
- Takes **interactive input** from the user (tenure, monthly charges, total charges, senior citizen status, contract type)
- Outputs a prediction: **"This customer is likely to CHURN"** or **"This customer is likely to STAY"**

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data loading, cleaning, transformation |
| scikit-learn | Model training, scaling, evaluation |

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bhanuprakashyirri/Project-ChurnGuard.git
   cd Project-ChurnGuard
   ```

2. Install dependencies:
   ```bash
   pip install pandas scikit-learn
   ```

---

## 🚀 Usage

Run each task in order:

```bash
# Step 1: Explore the data
python task1_load_explore.py

# Step 2: Clean the data
python task2_clean_data.py

# Step 3: Train and evaluate the model
python task3_train_model.py

# Step 4: Predict churn for a new customer
python task4_predict.py
```

When running `task4_predict.py`, you will be prompted to enter:
- Tenure (months)
- Monthly Charges
- Total Charges
- Senior Citizen status (1 = Yes, 0 = No)
- Contract type (0 = Month-to-month, 1 = One year, 2 = Two year)

---

## 📊 Dataset

The dataset (`churnguard_data.csv`) contains customer information including demographics, account details, and service usage. Key columns include:

- `tenure` — Number of months the customer has been with the company
- `MonthlyCharges` — Monthly billing amount
- `TotalCharges` — Total amount billed to date
- `Contract` — Contract type (Month-to-month, One year, Two year)
- `InternetService` — Internet service type (DSL, Fiber optic, No)
- `Churn` — Target label (Yes / No)

---

## 📈 Model

**Algorithm:** Logistic Regression  
**Features used in prediction:** `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `Contract`  
**Class balancing:** `class_weight='balanced'` to fairly handle the imbalanced churn distribution  
**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score

---

## 👤 Author

**Bhanu Prakash Yirri**  
GitHub: [@bhanuprakashyirri](https://github.com/bhanuprakashyirri)
