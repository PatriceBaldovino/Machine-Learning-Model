import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Load dataset
csv_path = './dataset/features_30_sec.csv'
df = pd.read_csv(csv_path)

# Display first few rows of the dataset
print(df.head())

# Data visualization: Count of samples per genre
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Number of Samples per Genre")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('./dataset/genre_distribution.png')
plt.show()

# Data preprocessing
X = df.drop(columns=['filename', 'label'])  # Features (exclude filename and label)
y = df['label']  # Target (genre labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Data visualization: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig('./dataset/confusion_matrix.png')
plt.show()

# Data visualization: Feature importance
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': clf.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances.head(10), x='importance', y='feature', palette='viridis')
plt.title("Top 10 Most Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig('./dataset/feature_importance.png')
plt.show()

# Save model, scaler, and label encoder
os.makedirs('./saved_models', exist_ok=True)
pickle.dump(clf, open('./saved_models/genre_model.pkl', 'wb'))
pickle.dump(scaler, open('./saved_models/scaler.pkl', 'wb'))
pickle.dump(le, open('./saved_models/label_encoder.pkl', 'wb'))
