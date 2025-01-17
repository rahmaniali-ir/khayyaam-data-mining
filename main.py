# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

# 1. Load dataset (replace with your dataset file path)
# For example, here we use the 'Iris' dataset
# Replace with your dataset path
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width',
           'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, header=None, names=columns)

# 2. Data Preprocessing
# 2.1 Cleaning the data: Handling missing values (if any)
print("Before cleaning:")
print(df.describe())

df.dropna(inplace=True)  # Drop rows with missing values if any

# 2.2 Feature Scaling: Standardization of numeric features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

# 2.3 Encoding categorical variables (if any)
# In this case, the target 'class' is categorical
df['class'] = df['class'].astype('category')
df['class'] = df['class'].cat.codes

# After preprocessing
print("\nAfter cleaning and scaling:")
print(pd.DataFrame(scaled_features, columns=[
      'sepal_length', 'sepal_width', 'petal_length', 'petal_width']).describe())

# 3. Split the data into training and testing sets
X = scaled_features
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 4. Choose and train models

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Support Vector Machine (SVM)
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# K-Means (Clustering)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

# 5. Evaluate models

# Random Forest Evaluation
print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred)}")
print(f"Classification Report:\n{classification_report(y_test, rf_pred)}")

# Decision Tree Evaluation
print("\nDecision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, dt_pred)}")
print(f"Classification Report:\n{classification_report(y_test, dt_pred)}")

# SVM Evaluation
print("\nSVM Performance:")
print(f"Accuracy: {accuracy_score(y_test, svm_pred)}")
print(f"Classification Report:\n{classification_report(y_test, svm_pred)}")

# Naive Bayes Evaluation
print("\nNaive Bayes Performance:")
print(f"Accuracy: {accuracy_score(y_test, nb_pred)}")
print(f"Classification Report:\n{classification_report(y_test, nb_pred)}")

# 6. Compare feature importances (for Random Forest)
feature_importances = rf_model.feature_importances_
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Random Forest')
plt.show()

# 7. Clustering Results: K-Means
print("\nK-Means Clustering Centers:")
print(kmeans.cluster_centers_)

# Visualize Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('K-Means Clustering (2D projection of first two features)')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.show()
