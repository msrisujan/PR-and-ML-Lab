import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv').drop('Id', axis=1)
#drop SepalLengthCm, SepalWidthCm
df = df.drop(['SepalLengthCm', 'SepalWidthCm'], axis=1)

setosa = np.array(df[df['Species'] == 'Iris-setosa'].drop('Species', axis=1))
versicolor = np.array(df[df['Species'] == 'Iris-versicolor'].drop('Species', axis=1))
virginica = np.array(df[df['Species'] == 'Iris-virginica'].drop('Species', axis=1))

train_setosa = setosa[:40,:]
train_versicolor = versicolor[:40,:]
train_virginica = virginica[:40,:]

def covariance(X):
    mean = np.mean(X,axis=0)
    X = X - mean
    return np.dot(X.T, X)/(X.shape[0]-1)
    
def find_case(w1, w2):
    cov_w1, cov_w2 = covariance(w1), covariance(w2)
    if (cov_w1 == cov_w2).all():
        identity = np.identity(w1.shape[1])
        if ((cov_w1[0, 0] * identity) == cov_w1).all():
            return 1
        else:
            return 2
    else:
        return 3

def g_of_x(X, apriori,case):
    cov = covariance(X)
    inv_cov = np.linalg.inv(cov)
    mean = np.mean(X,axis=0)
    cov_det = np.linalg.det(inv_cov)

    if case == 1:
        sigma_sq = cov[0, 0]
        A = np.zeros_like(inv_cov)
        B = mean.T / sigma_sq
        C = -0.5 * mean.T.dot(mean) / sigma_sq + np.log(apriori)
    elif case == 2:
        A = np.zeros_like(inv_cov)
        B = inv_cov.dot(mean)
        C = -0.5 * mean.T.dot(inv_cov).dot(mean) + np.log(apriori)
    elif case == 3:
        A = -0.5 * inv_cov
        B = inv_cov.dot(mean)
        C = -0.5 * mean.T.dot(inv_cov).dot(mean) - 0.5 * np.log(cov_det) + np.log(apriori)
    return lambda x: x.T @ A @ x + B.T @ x + C  
    
def discriminant_plot(g1, g2):
    x = np.linspace(-10,10,100)
    y = np.linspace(-10,10,100)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = g1(np.array([X[i,j],Y[i,j]])) - g2(np.array([X[i,j],Y[i,j]]))
    plt.contour(X, Y, Z, levels=[0])
    plt.axis([0, 8, -1, 4])
    
case = find_case(train_setosa, train_versicolor)
g1 = g_of_x(train_setosa,1/3,case)
g2 = g_of_x(train_versicolor,1/3,case)
discriminant_plot(g1, g2)
case = find_case(train_setosa, train_virginica)
g1 = g_of_x(train_setosa,1/3,case)
g3 = g_of_x(train_virginica,1/3,case)
discriminant_plot(g1, g3)
case = find_case(train_versicolor, train_virginica)
g2 = g_of_x(train_versicolor,1/3,case)
g3 = g_of_x(train_virginica,1/3,case)
discriminant_plot(g2, g3)
plt.scatter(setosa[:,0],setosa[:,1],color = "r", edgecolors="k")
plt.scatter(versicolor[:,0],versicolor[:,1],color = "g", edgecolors="k")
plt.scatter(virginica[:,0],virginica[:,1],color = "b",edgecolors="k")
plt.show()