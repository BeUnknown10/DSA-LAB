import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters
kernels = ['rbf']
gammas = [0.5]
Cs = [0.01, 1, 10]

# Initialize variables to store best accuracy and support vectors
best_accuracy = 0
best_support_vectors = None

# Iterate over different hyperparameters
for kernel in kernels:
    for gamma in gammas:
        for C in Cs:
            # Create the SVM classifier with the specified hyperparameters
            model = SVC(kernel=kernel, gamma=gamma, C=C, decision_function_shape='ovr')

            # Train the model
            model.fit(X_train, y_train)

            # Calculate accuracy on the test data
            accuracy = model.score(X_test, y_test)

            # Get the total number of support vectors
            total_support_vectors = np.sum(model.n_support_)

            # Update best accuracy and support vectors if current accuracy is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_support_vectors = total_support_vectors

            # Print the current hyperparameters and corresponding accuracy
            print(f"Kernel: {kernel}, Gamma: {gamma}, C: {C}, Accuracy: {accuracy}, Total Support Vectors: {total_support_vectors}")

# Print the best accuracy and total number of support vectors
print(f"\nBest Accuracy: {best_accuracy}")
print(f"Total Support Vectors for Best Model: {best_support_vectors}")
