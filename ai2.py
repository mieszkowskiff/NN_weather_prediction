from ai_np import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork2(NeuralNetwork):
    def __init__(self, structure):
        super().__init__(
            structure, 
            activation = "tanh",
            last_layer_activation = "tanh"
            )
    
    def normalize_train_X(self, X):
        self.mean = X.mean()
        self.std = X.std()
        return (X - self.mean) / self.std
    
    def normalize_train_Y(self, Y, u = -1, l = 1):
        self.max = Y.max()
        self.min = Y.min()
        return (Y - self.min) / (self.max - self.min) * (u - l) + l
    
    def normalize_test_X(self, X):
        return (X - self.mean) / self.std
    
    def normalize_test_Y(self, Y, u = -1, l = 1):
        return (Y - self.min) / (self.max - self.min) * (u - l) + l
    
    def train(self, X_dataframe, Y_dataframe, learning_rate=0.1, epochs=100):
        X = self.normalize_train_X(X_dataframe)
        Y = self.normalize_train_Y(Y_dataframe)
        X = X.values.transpose()
        Y = Y.values.transpose()
        print(X)
        print(Y)
        print(X.shape)
        print(Y.shape)
        super().perform_training(X, Y, learning_rate = learning_rate, number_of_epochs = epochs)
        
        


if __name__ == "__main__":
    np.random.seed(0)
    t = np.random.normal(0, 1, 100)
    df = pd.DataFrame({
        "x": t,
        "y": t * t + np.random.normal(0, 0.1, 100)
    })
    nn = NeuralNetwork2([2, 5, 1])
    nn.train(df, df[["y"]])

    
    