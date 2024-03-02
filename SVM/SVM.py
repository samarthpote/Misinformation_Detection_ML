from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
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



# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel type
    'gamma': ['scale', 'auto']  # Kernel coefficient
}

# Initialize the SVM classifier
svm_classifier = SVC()

# Initialize GridSearchCV with 5-fold cross-validation
grid_search_svm = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5)

# Fit GridSearchCV on the training data
grid_search_svm.fit(X_train_tfidf, y_train)

# Get the best parameters found by GridSearchCV
best_params_svm = grid_search_svm.best_params_
print("Best Parameters for SVM:", best_params_svm)

# Get the best estimator (model) found by GridSearchCV
best_svm_classifier = grid_search_svm.best_estimator_

# Evaluate performance on the validation set with the best estimator
validation_score_svm = grid_search_svm.best_score_
print("Validation Score with Best Parameters:", validation_score_svm)

from sklearn.metrics import accuracy_score, classification_report

# Use the best logistic regression model to predict labels on the validation set
validation_predictions = best_svm_classifier.predict(X_validation_tfidf)

# Calculate accuracy on the validation set
accuracy_validation = accuracy_score(y_validation, validation_predictions)
# print("Validation Accuracy:", accuracy_validation)

# Generate classification report on the validation set
report_validation = classification_report(y_validation, validation_predictions)
print("Validation Classification Report:")
print(report_validation)
with open('evaluate_model.pkl', 'wb') as f:
    pickle.dump(best_svm_classifier, f)
with open('SVM_model.pkl', 'wb') as f:
    pickle.dump(best_svm_classifier, f)