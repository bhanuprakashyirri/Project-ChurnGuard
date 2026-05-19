import pandas as pd

# load data
df = pd.read_csv('churnguard_data.csv')

df = df.drop(columns=['customerID'])
df = df.drop_duplicates()

# clean values
df['gender'] = df['gender'].str.strip()
df['PaymentMethod'] = df['PaymentMethod'].str.strip()

df['Churn'] = df['Churn'].str.strip().str.title()
df['PhoneService'] = df['PhoneService'].str.strip().str.title()
df['PaperlessBilling'] = df['PaperlessBilling'].str.strip().str.title()

contract_corrections = {
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
df['Contract'] = df['Contract'].replace(contract_corrections)

internet_corrections = {
    'dsl': 'DSL',
    'fiber optic': 'Fiber optic',
    'Fiber Optic': 'Fiber optic',
    'no': 'No',
    'None': 'No'
}
df['InternetService'] = df['InternetService'].replace(internet_corrections)

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

print("~~~ Cleaned Dataset Shape ~~~")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n")

print("~~~ Missing Value Counts After Cleaning ~~~")
print(df.isnull().sum())
print("\n")

print("~~~ First 5 Rows ~~~")
print(df.head())
print("\n")