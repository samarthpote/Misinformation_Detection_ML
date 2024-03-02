import pickle
from sklearn.metrics import accuracy_score, classification_report

# Load the K-means clustering model
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Load the cluster-label mapping dictionary
with open('cluster_labels.pkl', 'rb') as f:
    cluster_labels = pickle.load(f)
with open('../test_tfidf.pkl', 'rb') as f:
    X_test_tfidf = pickle.load(f)

with open('../y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
# Use the K-means model to predict cluster assignments for the test data
test_cluster_assignments = kmeans_model.predict(X_test_tfidf)  # Assuming X_test_tfidf is your test TF-IDF matrix

# Map cluster assignments to labels using the cluster-label mapping dictionary
test_predictions = [cluster_labels[cluster_id] for cluster_id in test_cluster_assignments]

# Calculate accuracy on the test set
accuracy_test = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", accuracy_test)

# Generate classification report on the test set
report_test = classification_report(y_test, test_predictions)
print("Test Classification Report:")
print(report_test)
