import numpy as np                      # Import NumPy for numerical operations and array handling
import matplotlib.pyplot as plt         # Import Matplotlib's pyplot for plotting
import pandas as pd                     # Import Pandas (not actually used later in this notebook)

from sklearn.datasets import make_blobs         # Utility to generate synthetic clustered data
from sklearn.model_selection import train_test_split  # Utility to split data into train/test sets
plt.style.use("seaborn")                        # Set plotting style to 'seaborn' for nicer visuals

# Step-1 Generate Toy(Dummy) Dataset

X, y = make_blobs(
    n_samples=2000,                      # Total number of data points
    n_features=2,                        # Number of features (dimensions) = 2
    cluster_std=3,                       # Standard deviation of each cluster (controls spread)
    centers=2,                           # Number of clusters/classes = 2 (binary classification)
    random_state=42                      # Seed for reproducibility
)
n_features = 2                           # Store number of original features (without bias term)
print(X.shape, y.shape)                  # Print shapes => (2000, 2) and (2000,) typically
print(y)                                 # Print labels array to see distribution of classes

# Step - 2 Visualise Dataset 
def visualise(X, y):
    """
    X: feature matrix (n_samples x 2)
    y: labels (n_samples,)
    This function plots a scatter of the two features, colored by class.
    """
    plt.scatter(
        X[:, 0],                         # First feature as x-axis
        X[:, 1],                         # Second feature as y-axis
        c=y,                             # Color points by class labels
        cmap="viridis"                   # Use 'viridis' color map
    )
    plt.show()                           # Display the plot

visualise(X, y)                          # Visualize the raw/unscaled dataset

# Step - 3 
def normalise(X):
    """
    Standardize features: (X - mean) / std.
    This gives zero mean and unit variance per feature.
    """
    u = X.mean(axis=0)                   # Mean of each column (feature)
    std = X.std(axis=0)                  # Std deviation of each column (feature)

    return (X - u) / std                 # Return standardized features

X = normalise(X)                         # Overwrite X with normalized version
visualise(X, y)                          # Visualize again after normalization

# Step - 4 Train Test Split 
XT, Xt, yT, yt = train_test_split(
    X,                                   # Full feature matrix
    y,                                   # Full label vector
    test_size=0.25,                      # 25% test, 75% train
    shuffle=False,                       # Do NOT shuffle: split as first 75% train, last 25% test
    random_state=0                       # Random state (irrelevant when shuffle=False)
)

print(XT.shape, yT.shape)                # Shape of training features and labels
print(Xt.shape, yt.shape)                # Shape of test features and labels
visualise(XT, yT)                        # Visualize training set
visualise(Xt, yt)                        # Visualize test set

# Model 
def sigmoid(z):
    """
    Sigmoid activation function: maps real numbers to (0,1).
    Used to convert linear scores to probabilities.
    """
    return 1 / (1 + np.exp(-z))          # Standard sigmoid formula

def hypothesis(X, theta):
    """
    Compute predicted probability: h_theta(X) = sigmoid(X @ theta).
    X: (m x (n+1)) with bias term included
    theta: ((n+1) x 1) parameter vector
    """
    return sigmoid(np.dot(X, theta))     # Linear combination then sigmoid

# Binary Cross Entropy 
def error(y, yp):
    """
    Binary cross-entropy (log loss):
    loss = -mean( y*log(yp) + (1-y)*log(1-yp) )
    y: true labels (m x 1)
    yp: predicted probabilities (m x 1)
    """
    loss = -np.mean(
        y * np.log(yp) +                 # Contribution when y = 1
        (1 - y) * np.log(1 - yp)         # Contribution when y = 0
    )
    return loss                          # Return scalar loss

def gradient(X, y, yp):
    """
    Gradient of the loss w.r.t theta for logistic regression.
    grad = -(1/m) * X^T (y - yp)
    """
    m = X.shape[0]                       # Number of samples (rows)
    grad = -(1 / m) * np.dot(X.T, (y - yp))  # Gradient vector ((n+1) x 1)
    return grad                          # Return gradient

def train(X, y, max_iters=100, learning_rate=0.1):
    """
    Train logistic regression using gradient descent.
    X: training data (WITH bias column added)
    y: labels reshaped to (m x 1)
    max_iters: number of gradient descent steps
    learning_rate: step size for gradient descent
    """
    # Randomly init theta 
    theta = np.random.randn(n_features + 1, 1)   # Random init for parameters (including bias)

    error_list = []                       # To track loss over iterations

    for i in range(max_iters):           # Loop for given number of iterations
        yp = hypothesis(X, theta)        # Predicted probabilities for all samples
        e = error(y, yp)                 # Compute loss
        error_list.append(e)             # Store loss for visualization
        grad = gradient(X, y, yp)        # Compute gradient
        theta = theta - learning_rate * grad  # Gradient descent update

    plt.plot(error_list)                 # Plot loss vs iteration
    return theta                         # Return final learned parameters

def predict(X, theta):
    """
    Predict class labels (0 or 1) based on probability >= 0.5.
    X: input data with bias column included
    """
    h = hypothesis(X, theta)             # Compute probabilities
    preds = np.zeros((X.shape[0], 1), dtype='int')  # Initialize predictions as 0
    preds[h >= 0.5] = 1                  # Set to 1 where probability >= 0.5

    return preds                         # Return predictions (m x 1)

def accuracy(X, y, theta):
    """
    Compute accuracy (%) = correctly classified / total * 100.
    NOTE: expects y to be (m x 1) to match preds.
    """
    preds = predict(X, theta)            # Get predictions
    return ((y == preds).sum()) / y.shape[0] * 100  # Accuracy in percent

def addExtraColumn(X):
    """
    Add a bias column of ones at the beginning IF not already present.
    X: (m x n) -> (m x (n+1)) when n == n_features.
    """
    if X.shape[1] == n_features:         # If current columns equal original feature count
        ones = np.ones((X.shape[0], 1))  # Column of ones (bias term)
        X = np.hstack((ones, X))         # Concatenate as first column

    return X                             # Return possibly-augmented X

XT = addExtraColumn(XT)                  # Add bias column to training data
print(XT)                                # Print to verify bias column added
Xt = addExtraColumn(Xt)                  # Add bias column to test data
print(Xt)                                # Print test features with bias
print(XT.shape)                          # Confirm new shape (m_train, n_features + 1)
yT = yT.reshape(-1, 1)                   # Reshape training labels to column vector
yt = yt.reshape(-1, 1)                   # Reshape test labels to column vector
print(yT.shape)                          # Print training label shape
print(yt.shape)                          # Print test label shape
theta = train(XT, yT,                    # Train model on training data
              max_iters=300,             # More iterations for better convergence
              learning_rate=0.2)         # Higher learning rate than default
theta                                   # In a notebook, last expression shows theta

# Decision Boundary Visualisation
XT = addExtraColumn(X)                   # BUGGY / INCONSISTENT:
                                        # Here you are overwriting XT using FULL X again
                                        # (all 2000 points, not just training). Also addExtraColumn
                                        # will add bias again if X has only n_features columns.
plt.scatter(
    XT[:, 1],                            # x-axis: first feature (after bias column)
    XT[:, 2],                            # y-axis: second feature
    c=yT,                                # ⚠ This is wrong: yT has 1500 rows, XT now has 2000 rows.
    cmap="viridis"                       # Color points by labels (mismatch in shapes)
)

x1 = np.linspace(-3, 3, 6)               # Create 6 x values between -3 and 3
x2 = -(theta[0][0] + theta[1][0] * x1) / theta[2][0]
                                        # Rearranged decision boundary:
                                        # theta0 + theta1*x1 + theta2*x2 = 0
                                        # => x2 = -(theta0 + theta1*x1)/theta2
plt.plot(x1, x2)                         # Plot the decision boundary line
plt.show()                               # Show scatter + decision boundary

# Predictions 
preds = predict(Xt, theta)               # Predict on test set

# Train Accuracy
accuracy(XT, yT, theta)                  # ⚠ Accuracy using XT (redefined) and yT (1500x1)
                                        # Shapes don’t match logically here; should use training subset only.

# Test Accuracy 
accuracy(Xt, yt, theta)                  # Accuracy on test data (this one is fine)

# ------ SkLearn Library -----
from sklearn.linear_model import LogisticRegression  # Import Sklearn's LogisticRegression model

# Create
X, y = make_blobs(                       # Generate a fresh dataset again
    n_samples=2000,
    n_features=2,
    cluster_std=3,
    centers=2,
    random_state=42
)

model = LogisticRegression()             # Create logistic regression model (default settings)

# Training 
model.fit(XT, yT)                        # ⚠ Here you are fitting on XT, yT from earlier,
                                        # not on the newly generated X, y.

# Predictions
model.predict(Xt)                        # Predict using Xt (from manual split)

# Scoring 
model.score(XT, yT)                      # Accuracy on training data (with bias column included)
model.score(Xt, yt)                      # Accuracy on test data

# ------ Multiclass Classification -----
X, y = make_blobs(                       # Generate multiclass dataset: 3 classes, 5 features
    n_samples=2000,
    n_features=5,
    cluster_std=3,
    centers=3,
    random_state=42
)
plt.scatter(
    X[:, 0],                             # Plot only first two dimensions
    X[:, 1],
    c=y,
    cmap='viridis'
)
plt.show()
print(np.unique(y, return_counts=True))  # Show unique labels and how many examples per class

model = LogisticRegression(multi_class='ovr')  # One-vs-rest logistic regression for multiclass
model.fit(X, y)                         # Fit on 5D multiclass data
model.predict(X)                        # Predict on training data
model.score(X, y)                       # Training accuracy
