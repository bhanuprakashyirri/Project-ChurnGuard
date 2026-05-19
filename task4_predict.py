import pandas as pd
from sklearn.linear_model import LogisticRegression

# load data
df = pd.read_csv('churnguard_data.csv')

# clean values
df = df.drop(columns=['customerID'])
df = df.drop_duplicates()

df['gender'] = df['gender'].str.strip()
df['PaymentMethod'] = df['PaymentMethod'].str.strip()

df['Churn'] = df['Churn'].str.strip().str.title()
df['PhoneService'] = df['PhoneService'].str.strip().str.title()
df['PaperlessBilling'] = df['PaperlessBilling'].str.strip().str.title()

contract_fixes = {
    'month to month': 'Month-to-month',
    'month-to-month': 'Month-to-month',
    'Monthly': 'Month-to-month',
    'One Year': 'One year',
    '1 year': 'One year',
    'one year': 'One year',
    'Two Year': 'Two year',
    '2 year': 'Two year',
    'two year': 'Two year'
}
df['Contract'] = df['Contract'].replace(contract_fixes)

internet_fixes = {
    'dsl': 'DSL',
    'fiber optic': 'Fiber optic',
    'Fiber Optic': 'Fiber optic',
    'no': 'No',
    'None': 'No'
}
df['InternetService'] = df['InternetService'].replace(internet_fixes)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df = df.drop(df[df['tenure'] <= 0].index)
df = df.drop(df[(df['MonthlyCharges'] < 10) | (df['MonthlyCharges'] > 200)].index)

# fill missing values
df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean())
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

median_tenure = round(df['tenure'].median())
df['tenure'] = df['tenure'].fillna(median_tenure)

internet_mode = df['InternetService'].mode()[0]
df['InternetService'] = df['InternetService'].fillna(internet_mode)

# encode columns
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['Contract'] = df['Contract'].map({
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
})

features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Contract']
X = df[features]
y = df['Churn']

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Model trained successfully!\n")

# take input
user_tenure = float(input("Enter tenure (months): "))
user_monthly = float(input("Enter Monthly Charges: "))
user_total = float(input("Enter Total Charges: "))
user_senior = int(input("Senior Citizen? (1 = Yes, 0 = No): "))
user_contract = int(input("Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): "))

# make prediction
input_df = pd.DataFrame([[
    user_tenure, 
    user_monthly, 
    user_total, 
    user_senior, 
    user_contract
]], columns=features)

prediction = model.predict(input_df)[0]

print("\n--- Prediction Result ---")
if prediction == 1:
    print("Prediction: This customer is likely to CHURN.")
else:
    print("Prediction: This customer is likely to STAY.")