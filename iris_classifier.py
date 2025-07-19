# iris_classifier.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("C:\\Users\\Lamyaa\\Downloads\\Iris.csv")

# Drop the 'Id' column
df.drop("Id", axis=1, inplace=True)

# Show basic info
print("First 5 rows of the dataset:")
print(df.head())

# Visualize pairplot
sns.pairplot(df, hue='Species')
plt.title("Pairplot of Iris Dataset")
plt.show()

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
