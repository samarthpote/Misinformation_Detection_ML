from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle

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

# print(y_train[:10])

param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # Varying number of neighbors
    'weights': ['uniform', 'distance'],  # Different weight functions
    'algorithm': ['brute'],  # Use only the 'brute' algorithm
    'p': [1, 2]  # Power parameter for the Minkowski metric (1 for Manhattan distance, 2 for Euclidean distance)
}


# Initialize the GridSearchCV object for KNN
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search_knn.fit(X_train_tfidf, y_train)

# Get the best parameters found by GridSearchCV
best_params_knn = grid_search_knn.best_params_
print("Best Parameters for KNN:", best_params_knn)
best_params_knn = {'algorithm': 'brute', 'n_neighbors': 7, 'p': 2, 'weights': 'distance'}
# best_params_knn = {'algorithm': 'brute', 'n_neighbors': 9, 'p': 2, 'weights': 'uniform'}

knn_clf = KNeighborsClassifier(**best_params_knn)

# Train the KNN classifier
knn_clf.fit(X_train_tfidf, y_train)
# Generate predictions on the test set using the trained KNN classifier
val_predictions_knn = knn_clf.predict(X_validation_tfidf)

# Generate classification repor
report = classification_report(y_validation, val_predictions_knn)
print("Classification Report:")
print(report)
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_clf, f)
