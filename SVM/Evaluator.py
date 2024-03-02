import pickle
from sklearn.metrics import accuracy_score, classification_report

# Load the KNN model from the pickled file
with open('SVM_model.pkl', 'rb') as f:
    the_model = pickle.load(f)

with open('../test_tfidf.pkl', 'rb') as f:
    X_test_tfidf = pickle.load(f)

with open('../y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Generate predictions on the test set using the loaded model
test_predictions = the_model.predict(X_test_tfidf)

# Generate classification report
report = classification_report(y_test, test_predictions)
print("Classification Report:")
print(report)
