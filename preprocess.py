import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Select features
    feature_to_use = [
    'TotalCharges', 'MonthlyCharges', 'tenure', 'Contract', 'PaymentMethod', 'OnlineSecurity',
    'TechSupport', 'gender', 'InternetService', 'OnlineBackup', 'PaperlessBilling'
    ]

    # Subset the dataframe
    X = df[feature_to_use].copy()
    y = df['Churn'].copy()

    # Identify categorical columns
    categorical_features = X.select_dtypes(include='object').columns

    # Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y