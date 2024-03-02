from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Read the TF-IDF matrices
with open('../train_tfidf.pkl', 'rb') as f:
    X_train_tfidf = pickle.load(f)

with open('../validation_tfidf.pkl', 'rb') as f:
    X_validation_tfidf = pickle.load(f)
    
# Read the TfidfVectorizer instance
with open('../tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Read the labels
with open('../y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('../y_validation.pkl', 'rb') as f:
    y_validation = pickle.load(f)

with open('../y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)





# Step 1: Feature Extraction with K-means
kmeans = KMeans(
    n_clusters=5,  # Set the number of clusters
    init='k-means++',  # Use smart initialization
    n_init=5,  # Run the KMeans algorithm 10 times with different centroid seeds
    max_iter=300,  # Maximum number of iterations for each initialization
    tol=1e-4,  # Tolerance to declare convergence
    random_state = 42
)
cluster_assignments = kmeans.fit_predict(X_train_tfidf)  # Assuming X_train_tfidf is your TF-IDF matrix

# Step 2: Map Clusters to Labels 
cluster_labels = {}
for cluster_id in range(kmeans.n_clusters):
    cluster_data_indices = (cluster_assignments == cluster_id)
    cluster_labels[cluster_id] = Counter(y_train[cluster_data_indices]).most_common(1)[0][0]

# Step 3: Evaluate Performance
validation_cluster_assignments = kmeans.predict(X_validation_tfidf)  # Assuming X_validation_tfidf is your validation TF-IDF matrix
validation_predictions = [cluster_labels[cluster_id] for cluster_id in validation_cluster_assignments]

# Calculate accuracy on the validation set
accuracy_validation = accuracy_score(y_validation, validation_predictions)
# print("Validation Accuracy:", accuracy_validation)

# Generate classification report on the validation set
report_validation = classification_report(y_validation, validation_predictions)
print("Validation Classification Report:")
print(report_validation)

# Save the K-means clustering model
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Save the cluster-label mapping dictionary
with open('cluster_labels.pkl', 'wb') as f:
    pickle.dump(cluster_labels, f)