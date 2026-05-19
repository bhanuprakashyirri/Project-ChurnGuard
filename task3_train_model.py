import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

# encode columns
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = ['gender', 'PhoneService', 'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop(columns=['Churn'])
y = df['Churn']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
print("~~~ Training Logistic Regression Model ~~~")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n~~~ Model Evaluation ~~~")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))