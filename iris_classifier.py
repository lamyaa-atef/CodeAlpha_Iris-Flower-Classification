# iris_classifier.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("Iris.csv")

# Show basic info
print("First 5 rows of the dataset:")
print(df.head())

# Drop the 'Id' column
df.drop("Id", axis=1, inplace=True)

# EDA Checks to make sure data is clean

print("Data Info:")
print(df.info())
print("\nChecking for Missing Values:") # There are no missing values in the dataset
print(df.isnull().sum())

print("\nChecking for Zeros in Numeric Columns:") # There are no zeros in the dataset, but we check just in case     
print((df.select_dtypes(include='number') == 0).sum())

print("\nChecking for Duplicates:") # The data is small so we won't delete duplicates
print(f"Duplicates: {df.duplicated().sum()}")

print("\nData Types:")
print(df.dtypes)

# Visualize pairplot
sns.pairplot(df, hue='Species')
plt.savefig("Pairplot_of_Iris_Dataset.png")
plt.close()

# Split data into features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))