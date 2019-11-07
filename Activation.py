import numpy as np
class Activation(object):
    #contains many activation functions and their derivatives
    @staticmethod
    def tanh(X):
        return np.tanh(X)

    @staticmethod
    def tanh_prime(X):
        return 1 - Activation.tanh(X)**2    

    @staticmethod     
    def relu(X):
        X_copy=np.copy(X)
        X_copy[X_copy<= 0] = X_copy[X_copy<=0] * 0.01
        return X_copy  

    @staticmethod
    def relu_prime(X,alpha=0.01):
        X_copy=np.copy(X)
        X_copy[X_copy > 0] = 1
        X_copy[X_copy < 0] = 0.01
        return X_copy

    @staticmethod
    def sigmoid(X):
        return 1/(1+np.exp(-X))

    @staticmethod
    def sigmoid_prime(X):
          return Activation.sigmoid(X) * (1 - Activation.sigmoid(X))  
    
    @staticmethod
    def softmax(X,esp=1e-10):
        X_copy = X - np.max(X)
        return np.exp(X_copy) / np.sum(np.exp(X_copy)) 

    @staticmethod
    def softmax_prime(X,esp=1e-10):
        X_copy = X - np.max(X)
        S = np.sum(np.exp(X_copy))
        return  np.array([np.exp(x+esp) * (S - x) for x in X_copy.flatten()])/np.square(S+esp)

    @staticmethod 
    def activate(matrix,type:str):
        if type == "relu":
            return Activation.relu(matrix)
        elif type == "sigmoid":
            return Activation.sigmoid(matrix)
        elif type == "tanh":
            return Activation.tanh(matrix) 
        elif type == "softmax":
            return Activation.softmax(matrix)    

    @staticmethod 
    def derivative(matrix,type:str):
        if type == "relu":
            return Activation.relu_prime(matrix)
        elif type == "sigmoid":
            return Activation.sigmoid_prime(matrix)
        elif type == "tanh":
            return Activation.tanh_prime(matrix)
        elif type == "softmax":
            return Activation.softmax_prime(matrix)   