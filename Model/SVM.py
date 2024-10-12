import numpy as np

class SVC():

    # initiating the hyperparameter
    '''
        learning_rate: The learning speed of the algorithm. This is the step size in updating the weights.
        no_of_iterations: The number of loops to update weights and bias.
        lambda_parameter: Regularization, which helps prevent overfitting by controlling the magnitude of the weights.
    '''
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter



    # Fitting the dataset to SVC
    def fit(self, X, y):
        # m --> number of Data points --> number of rows
        # n --> number of input features --> number of columns
        self.m, self.n = X.shape

        # Initiating the weight value and bias values.
        # Initialize the initial weighted vector with a value of 0 for all features.
        # self.w is the weight.
        self.w = np.zeros(self.n)

        # Initiates an initial bias value of 0.
        # self.b is the bias.
        self.b = 0

        # Store the training dataset into class variable for use during training
        self.X = X
        self.y = y

        # implementing Gradient Descent algorithm for Optimization
        # During each iteration, it calls the update_weights() function to compute and adjust the parameters of the model
        for i in range(self.no_of_iterations):
            self.update_weights()




    # function for updating the weight and bias value
    def update_weights(self):

        # label encoding
        # convert the y_label to {-1, 1} instead of {0,1} as the SVM requires the labels to be +1 and -1 for training
        y_label = np.where(self.y <= 0, -1, 1)

        # gradient (dw, db)
        # index: This is the index of the current element in the self.X array. It starts at 0
        # x_i: This is the data point that corresponds to that metric
        for index, x_i in enumerate(self.X):

            # (np.dot(x_i, self.w) - self.b): Calculate the decision boundary value
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

            # If this condition is true >= 1
            if (condition == True):
                # This is the derivative of the normalization component (L2 regularization) in the loss function
                dw = 2 * self.lambda_parameter * self.w
                db = 0

            else:
                # The weighted update formula includes both the normalization
                # component and the classification error correction component
                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
                db = y_label[index]

            # Use gradient descent to update weights and bias.
            # The w weights and b bias are adjusted based on the gradient and the learning_rate
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db




    # Predict the label for a given input value
    def predict(self, X):
        # Calculate the value of the decisive function for each data point in the new set X
        y_hat = np.dot(X, self.w) - self.b
        return y_hat


