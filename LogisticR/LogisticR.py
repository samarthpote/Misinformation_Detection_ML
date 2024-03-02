from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

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
    'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization parameter
    'penalty': ['l1', 'l2'],  # Penalty term
    'solver': ['liblinear'],  # Algorithm to use in optimization problem
    'max_iter': [100, 200, 300]  # Maximum number of iterations
}

# Initialize the logistic regression classifier
logistic_regression = LogisticRegression()

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, cv=5)

# Fit GridSearchCV on the training data
grid_search.fit(X_train_tfidf, y_train)

# Get the best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters for Logistic Regression:", best_params)
# Get the best estimator (model) found by GridSearchCV
best_logistic_regression = grid_search.best_estimator_

# Evaluate performance on the validation set with the best estimator
validation_score = grid_search.best_score_
print("Validation Score with Best Parameters:", validation_score)

from sklearn.metrics import accuracy_score, classification_report

# Use the best logistic regression model to predict labels on the validation set
validation_predictions = best_logistic_regression.predict(X_validation_tfidf)



# Generate classification report on the validation set
report_validation = classification_report(y_validation, validation_predictions)
print("Validation Classification Report:")
print(report_validation)
with open('Logistic_r.pkl', 'wb') as f:
    pickle.dump(best_logistic_regression, f)