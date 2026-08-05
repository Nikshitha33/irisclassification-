import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split dataset into training (80%) and testing (20%) subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Construct a Scikit-Learn Pipeline combining StandardScaler and KNeighborsClassifier
# StandardScaler normalizes features so each has mean=0 and variance=1.
# KNeighborsClassifier utilizes K=5 with distance-weighted voting.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
])

# 4. Train the pipeline
pipeline.fit(X_train, y_train)

# 5. Predict on test set
y_pred = pipeline.predict(X_test)

# 6. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)

# Print metrics to console
print("================ MODEL EVALUATION METRICS ================")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f} (weighted)")
print(f"Recall:    {recall:.4f} (weighted)")
print(f"F1 Score:  {f1:.4f} (weighted)")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("==========================================================")

# 7. Serialize and save the entire pipeline to model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Pipeline saved successfully as 'model.pkl'!")
