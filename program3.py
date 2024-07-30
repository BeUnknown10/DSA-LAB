from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with standardization and logistic regression 
model = make_pipeline(StandardScaler(), LogisticRegression(C=1e4))

# Train the model
model.fit(X_train, y_train)

# Report the training accuracy
training_accuracy = model.score(X_train, y_train) 
print(f"Training Accuracy: {training_accuracy}")

# Report the testing accuracy
testing_accuracy = model.score(X_test, y_test)
print(f"Testing Accuracy: {testing_accuracy}")
