#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


def accuracy(y_test, y_pred):
    return np.sum(y_pred == y_test) / len(y_test)


# In[3]:


def decision_boundary(epoch, X, y, weights, bias):
        plt.figure()
        plt.rcParams['figure.figsize'] = [4, 3.2]
        plt.scatter(X[:, 0], X[:, 1], c = y)
        x1 = np.linspace(-0.2, 1.2, 100)
        x2 = -(weights[0] * x1 + bias) / weights[1]
        plt.plot(x1, x2, 'r')
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        plt.title("Interation: " + str(epoch))
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show(block = False)
        plt.pause(0.1)
        plt.close()


# ### Perceptron

# In[4]:


class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def forward_propagation(self, x):
        return self.activation(np.dot(self.weights, x) + self.bias)
    
    def back_propagation(self, x, y, y_hat):
        self.weights += self.learning_rate * (y - y_hat) * x
        self.bias += self.learning_rate * (y - y_hat)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.converged = False


        for epoch in range(1, self.max_epochs+1):
            errors = 0
            for i in range(len(X)):
                y_hat = self.forward_propagation(X[i])
                self.back_propagation(X[i], y[i], y_hat)
                errors += int(y[i] != y_hat)
            if errors == 0:
                self.converged = True
                break
            print(f"Iteration: {epoch} | Weights: {self.weights} | Bias: {self.bias}")
            decision_boundary(epoch, X, y, self.weights, self.bias)

    def predict(self, X):
        y_pred = []
        for x in X:
            y = self.forward_propagation(x)
            y_pred.append(y)
        return np.array(y_pred)
        


# In[5]:


# X and Y for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])


# In[6]:


perceptron = Perceptron()
perceptron.fit(X, y)


# In[7]:


y_pred = perceptron.predict(X)
print(f"Predicted: {y_pred}")
print(f"Actual: {y}")


# In[8]:


acc = accuracy(y, y_pred)
print(f"Accuracy percentage: {acc * 100}%")


# ### LinearSVM

# In[9]:


class LinearSVM:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        return 1 if x >= 0 else 0
        

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for epoch in range(1, self.max_epochs+1):
            for i in range(len(X)):
                if y[i] * (np.dot(self.weights, X[i]) + self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * 1 / epoch * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * 1 / epoch * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]
            print(f"Iteration: {epoch} | Weights: {self.weights} | Bias: {self.bias}")
            decision_boundary(epoch, X, y, self.weights, self.bias)

    def predict(self, X):
        y_pred = []
        for x in X:
            y = self.activation(np.dot(self.weights, x) + self.bias)
            y_pred.append(y)
        return np.array(y_pred)


# In[10]:


svm = LinearSVM(max_epochs=100)
svm.fit(X, y)


# In[11]:


y_pred = svm.predict(X)
print(f"Predicted: {y_pred}")
print(f"Actual: {y}")


# In[12]:


acc = accuracy(y, y_pred)
print(f"Accuracy percentage: {acc * 100}%")

