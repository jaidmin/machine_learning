import numpy as np
import scipy.io
#from scipy import io
import matplotlib.pyplot as plt
#%matplotlib inline
import random
import math
import io
import scipy.optimize, scipy.special
import sklearn.preprocessing
import sklearn.linear_model 
import pandas as pd
from scipy.special import expit
from scipy.sparse import issparse, csr_matrix
from sklearn.linear_model import *
from sympy import *
import numdifftools as nd

vars = {}
a = scipy.io.loadmat("ex3data1.mat",vars)

X = vars['X']
y = vars['y']
np.place(y,y==10,0)
m = len(X)
theta = np.zeros((X.shape[1]+1,1))
X = np.concatenate((np.ones((m,1)),X),axis=1)

def predict(weights,dataset):
    if(len(weights.shape)==1):
        weights = weights[:,np.newaxis]
    #print("weights.shape=",weights.shape)
    #print("dataset.shape=",dataset.shape)
    prediction = scipy.special.expit(np.dot(dataset,weights))
    return prediction

def cost(weights,dataset,labels,gamma=0):
    if(len(weights.shape)==1):
        weights = weights[:,np.newaxis]
    regularizer = (gamma/(2*m))*np.sum(np.delete(labels,0,0))
    prediction = predict(weights,dataset)
    cost = -1/m * np.sum(np.dot(labels.T,np.log(prediction))+np.dot((1-labels).T,np.log(1-prediction))) +regularizer
    #print(cost)
    return cost

def one_hot_encoding(labels,class_nr):
    new_labels = np.copy(labels)
    np.place(new_labels,new_labels==class_nr,1) 
    np.place(new_labels,new_labels!=1,0)
    return new_labels

def gradient(weights,dataset,labels,gamma=0):
    if(len(weights.shape)==1):
        weights = weights[:,np.newaxis]
    regularizer = (gamma/m) * (np.insert(np.delete(theta,0,0),0,0,0))
    prediction = predict(weights,dataset)
    grad = 1/m*np.dot(X.T,(prediction-labels)) +regularizer
    return np.ndarray.flatten(grad)

def hessian(weights,dataset,labels,gamma=0.1):
    W = predict(weights,dataset) * (1-predict(weights,dataset))
    W = np.diag(W.flatten())
    return np.dot(np.dot(-dataset.T,W),dataset)

def minimize(weights,dataset,labels):
    return scipy.optimize.minimize(cost,weights,method='trust-ncg',jac=gradient,hess=hessian,args=(dataset,labels),tol=0.00001)


def train(dataset,labels,class_nr):
    new_labels = one_hot_encoding(labels,class_nr)
    theta = np.zeros((dataset.shape[1],1))
    return minimize(weights = theta,dataset=dataset,labels=new_labels)

print("hello i got to the for loop")

print(minimize(theta,X,one_hot_encoding(y,1)))

'''
for i in range(0,10):

    if i == 0:
        print("first iterations")
        all_theta = train(X,y,i)#['x']
        all_theta = all_theta[:,np.newaxis]
    else:
        print(i)
        curr_theta = train(X,y,i)#['x']
        curr_theta = curr_theta[:,np.newaxis]
        all_theta = np.concatenate((all_theta,curr_theta),axis=1)

p = predict(all_theta,X)

p.shape

y_hat = np.argmax(p,axis=1)
y_hat = y_hat[:,np.newaxis]

difference = y - y_hat

errors = 0
kind_of_error = []

for i in range(0,len(y)-1):
    if difference[i] != 0:
        errors +=1
        kind_of_error.append((y_hat[i],y[i]))

(len(y)-errors) / len(y)
'''
   
