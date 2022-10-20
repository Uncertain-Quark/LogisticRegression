# Date: 19th Oct, 2022
# Author: Aditya Kommineni
import numpy as np

class binarylogisticregression:
    def __init__(self, alpha, dim_input, dim_output, n_iters: int = 500, random_seed: int = 42):
        self.alpha = alpha
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_iters = n_iters
        
        # Setting the seed to maintain the initialization
        np.random.seed(random_seed)

        # Initializing the weights
        self.weights = np.random.randn(self.dim_input, self.dim_output)
        # self.weights = np.zeros((self.dim_input, self.dim_output))
        # Initializing the bias
        self.bias = np.random.randn(self.dim_output,1)

    def _sigmoid(self, x):
        epsilon = 1
        return  np.exp(1)/(np.exp(1) + np.exp(1-x))

    def _linear(self, x):
        return np.matmul(x, self.weights) + self.bias

    def _logloss(self, a, y_true):
        epsilon = 1e-20
        loss = -(1/a.shape[0])*(np.sum(y_true * np.log(a + epsilon) + (1-y_true) * np.log(1-a + epsilon)))
        return loss

    def _gradientweight(self, x, a, y_true):
        # TODO compute the weights gradient
        gradient_weights = (np.matmul((a-y_true).T, x)).T*(1/x.shape[0])
        return gradient_weights
    
    def _gradientbias(self, a, y_true):
        # TODO compute the bias gradient
        gradient_bias = np.sum(a-y_true).reshape(1,1)
        return gradient_bias

    def train(self, x, y_true):
        '''Function to train using a given input'''
        loss_list = []
        weights_list = []

        for i in range(self.n_iters):
            # Computing the sigmoid output given the input
            z = self._linear(x)
            a = self._sigmoid(z)

            # Computing the log loss
            loss = self._logloss(a, y_true)
            print("Iteration {} Loss {}".format(i, loss))

            # Train accuracy
            y_train_pred = (a > 0.5).astype(int)
            print("Accuracy Train : ", np.sum(y_train_pred == y_true)/60000)

            # Computing the gradients
            gradient_weights = self._gradientweight(x, a, y_true)
            gradient_bias = self._gradientbias(a, y_true)

            # Updating the weights using the gradients
            self.weights = self.weights - self.alpha * gradient_weights
            self.bias = self.bias - self.alpha * gradient_bias

            loss_list.append(loss)
            weights_list.append(self.weights.copy())

        return weights_list, loss_list
    
    def inference(self, x_test, y_test):
        # Test Accuracy
        y_test_pred = (self._sigmoid(self._linear(x_test)) > 0.5).astype(int)
        print("Accuracy Test : ", np.sum(y_test_pred == y_test)/10000)
