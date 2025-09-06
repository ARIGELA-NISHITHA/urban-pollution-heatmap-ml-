import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample loading (replace with actual dataset)
df = pd.read_csv("urban_pollution_data.csv")

# Feature engineering
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month

# Drop irrelevant columns
df = df.drop(columns=["datetime", "station_id"])

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

# Features and labels
X = df.drop(columns=["PM2.5"])  # target column
y = df["PM2.5"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
