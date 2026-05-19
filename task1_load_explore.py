import pandas as pd

# load data
df = pd.read_csv(r"C:\Users\BHANU PRAKASH\OneDrive\Documents\churnguard_data.csv")

print("~~~ Dataset Shape ~~~")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}\n")

print("~~~ First 5 Rows ~~~")
print(df.head())
print("\n")

print("~~~ Column Names and Data Types ~~~")
df.info()
print("\n")

print("~~~ Missing Values Count ~~~")
print(df.isnull().sum())
print("\n")

print("~~~ Duplicate Rows Count ~~~")
print(f"Total duplicate rows: {df.duplicated().sum()}\n")

print("~~~ Value Counts of 'Churn' Column ~~~")
print(df['Churn'].value_counts())
print("\n")

print("~~~ Unique Values in 'Contract' Column ~~~")
print(df['Contract'].unique())
print("\n")