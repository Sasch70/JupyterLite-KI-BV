# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:20:46 2022

@author: stefan.kray
"""

import numpy as np
hidden_weights=np.array([[20,  -20],
                         [20, -20]])
hidden_b=np.array([-10,+30])
out_weights=np.array([20,20])
out_b=np.array([-30])

X = np.array([[0,0], [0,1], [1,0], [1,1]])
print("Eingabe:")
print(X)

def threshold(x): 
    return (x>0).astype(int)
def sigmoid(x): 
    return np.round(1/(1+ np.exp(-x)),2)


# --- code ab hier ---
net = np.dot(X, hidden_weights)
net_b = net+hidden_b
h=sigmoid(net_b)
result=np.dot(h,out_weights)+out_b
print("Ausgabe:")
print(sigmoid(result))