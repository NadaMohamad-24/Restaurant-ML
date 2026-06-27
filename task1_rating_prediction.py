# ============================================
# Task 1: Predict Restaurant Ratings
# Cognifyz Technologies - Machine Learning Internship
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load Dataset
df = pd.read_csv("Dataset.csv")

# Display the first five rows
print(df.head())

# =====================================
# Explore the Dataset
# =====================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nFirst 5 Rows:")
print(df.head())

# =====================================
# Data Cleaning
# =====================================

print("\nRemoving unnecessary columns...")

columns_to_drop = [
    "Restaurant ID",
    "Restaurant Name",
    "Address",
    "Locality",
    "Locality Verbose"
]

df = df.drop(columns=columns_to_drop)

print("\nRemaining Columns:")
print(df.columns)

print("\nDataset Shape After Cleaning:")
print(df.shape)

# =====================================
# Encode Categorical Columns
# =====================================

print("\nEncoding categorical columns...")

encoder = LabelEncoder()

for column in df.select_dtypes(include="object").columns:
    df[column] = encoder.fit_transform(df[column])

print("\nEncoded Dataset:")
print(df.head())

# =====================================
# Select Features and Target
# =====================================

print("\nSelecting Features and Target...")

X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# =====================================
# Split Dataset
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Set:", X_train.shape)
print("Testing Set:", X_test.shape)

# =====================================
# Train Linear Regression Model
# =====================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# =====================================
# Make Predictions
# =====================================

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])


# =====================================
# Evaluate Model
# =====================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== Model Evaluation ==========")
print(f"Mean Absolute Error : {mae:.4f}")
print(f"Mean Squared Error  : {mse:.4f}")
print(f"R2 Score            : {r2:.4f}")

# =====================================
# =====================================
# Data Visualization
# =====================================

# 1. Distribution of Ratings
plt.figure(figsize=(8,5))
sns.histplot(df["Aggregate rating"], bins=20, kde=True)

plt.title("Distribution of Restaurant Ratings")
plt.xlabel("Aggregate Rating")
plt.ylabel("Count")

plt.savefig("rating_distribution.png")
plt.close()


# 2. Votes vs Rating
plt.figure(figsize=(8,5))
sns.scatterplot(x="Votes", y="Aggregate rating", data=df)

plt.title("Votes vs Aggregate Rating")
plt.xlabel("Votes")
plt.ylabel("Aggregate Rating")

plt.savefig("votes_vs_rating.png")
plt.close()


# 3. Rating Categories
plt.figure(figsize=(8,5))
sns.countplot(x="Rating text", data=df)

plt.title("Restaurant Rating Categories")
plt.xticks(rotation=30)

plt.savefig("rating_categories.png")
plt.close()


# 4. Correlation Heatmap
plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")
plt.close()

print("\nVisualization images saved successfully!")

# =====================================
# Model Comparison
# =====================================

print("\n========== Model Comparison ==========")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, mse, r2])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "MSE", "R2 Score"]
)

print(results_df)

# =====================================
# Compare Models
# =====================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("Model Comparison (R2 Score)")
plt.xticks(rotation=15)

plt.savefig("model_comparison.png")
plt.close()

print("\nModel comparison chart saved!")