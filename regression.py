# -*- coding: utf-8 -*-

'''
COMP551 PROJECT 1

LINEAR REGRESSION

Vincent Bazinet
Gabriel Desrosiers-Grégoire
Kevin Ma

***MAKE SURE THAT ALL THE FILES FOR THE PROJECT ARE IN THE SAME DIRECTORY

***COMMENT OUT WHATEVER LINE YOU DONT WANT TO RUN

'''

import numpy as np
import time

#load pre-processed data (COMMENT OUT THE FOLLOWING IMPORT STATEMENT IF DATA IS ALREADY LOADED)
import preprocessing as pp

 #Initialize the data set 
print("Initializing the datasets... ( 3 vs 63 vs 163 features)")
X_3 = pp.X_train[:,:3]
X_63 = pp.X_train[:,:63]
X_163 = pp.X_train[:,:163]

X_valid_3 = pp.X_valid[:,:3]
X_valid_63 = pp.X_valid[:,:63]
X_valid_163 = pp.X_valid[:,:163]

Y = pp.Y_train
Y = Y.reshape((10000, 1))

Y_valid = pp.Y_valid
Y_valid = Y_valid.reshape((1000, 1))

#add a dummy feature
dummy = np.zeros((10000, 1))
dummy = dummy + 1.0000

dummy_v = np.zeros((1000, 1))
dummy_v = dummy_v + 1.0000

X_3 = np.append(X_3, dummy, axis=1)
X_63 = np.append(X_63, dummy, axis=1)
X_163 = np.append(X_163, dummy, axis=1)

X_valid_3 = np.append(X_valid_3, dummy_v, axis=1)
X_valid_63 = np.append(X_valid_63, dummy_v, axis=1)
X_valid_163 = np.append(X_valid_163, dummy_v, axis=1)

###LINEAR REGRESSION --- CLOSED-FORM SOLUTION

###EXPERIMENT NUMBER 1 (RUNTIME)

start_time_closed = (time.time()) * 1000

w_closed_3 = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_3), X_3)), np.matmul(np.transpose(X_3), Y))

end_time_closed = (time.time()) * 1000
time_closed = end_time_closed - start_time_closed
print(time_closed)

###EXPERIMENT NUMBER 2 (3 vs 63 vs 163 features)
#compute coefficients (w)
w_closed_3 = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_3), X_3)), np.matmul(np.transpose(X_3), Y))
w_closed_63 = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_63), X_63)), np.matmul(np.transpose(X_63), Y))
w_closed_163 = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_163), X_163)), np.matmul(np.transpose(X_163), Y))

#compute mean squared error (MSE) on training set
MSE_closed_train_3 = np.matmul(np.transpose(np.subtract(Y,(np.matmul(X_3, w_closed_3)))),np.subtract(Y,(np.matmul(X_3, w_closed_3))))
MSE_closed_train_3 = MSE_closed_train_3/10000.000

MSE_closed_train_63 = np.matmul(np.transpose(np.subtract(Y,(np.matmul(X_63, w_closed_63)))),np.subtract(Y,(np.matmul(X_63, w_closed_63))))
MSE_closed_train_63 = MSE_closed_train_63/10000.000

MSE_closed_train_163 = np.matmul(np.transpose(np.subtract(Y,(np.matmul(X_163, w_closed_163)))),np.subtract(Y,(np.matmul(X_163, w_closed_163))))
MSE_closed_train_163 = MSE_closed_train_163/10000.000

#compute mean squared error (MSE) on validation set
MSE_closed_val_3 = np.matmul(np.transpose(np.subtract(Y_valid,(np.matmul(X_valid_3, w_closed_3)))),np.subtract(Y_valid,(np.matmul(X_valid_3, w_closed_3))))
MSE_closed_val_3 = MSE_closed_val_3/1000.000

MSE_closed_val_63 = np.matmul(np.transpose(np.subtract(Y_valid,(np.matmul(X_valid_63, w_closed_63)))),np.subtract(Y_valid,(np.matmul(X_valid_63, w_closed_63))))
MSE_closed_val_63 = MSE_closed_val_63/1000.000

MSE_closed_val_163 = np.matmul(np.transpose(np.subtract(Y_valid,(np.matmul(X_valid_163, w_closed_163)))),np.subtract(Y_valid,(np.matmul(X_valid_163, w_closed_163))))
MSE_closed_val_163 = MSE_closed_val_163/1000.000

###LINEAR REGRESSION --- GRADIENT DESCENT

###EXPERIMENT NUMBER 1 (RUNTIME)

start_time_descent = (time.time()) * 1000

#Initial weights
w_descent = np.zeros((4,1))

#Hyperparameters:
n_0 = 0.0000065
epsilon = 0.001

i = 1

while True:  
    
    #PRINT THE MSE for the current coefficients
    #RSS_descent = np.matmul(np.transpose(np.subtract(Y,(np.matmul(X, w_descent)))),np.subtract(Y,(np.matmul(X, w_descent))))
    #MSE_descent = RSS_descent/10000.000
    #print(MSE_descent)
    
    alpha = n_0
    
    #compute the new coefficients
    derived_RSS = 2 * np.subtract(np.matmul(np.matmul(np.transpose(X_3), X_3), w_descent), np.matmul(np.transpose(X_3), Y ))
    w_descent2 = w_descent - (alpha * derived_RSS)
    
    #stop when descent is completed
    if (np.linalg.norm((w_descent - w_descent2))) < epsilon:
        break
    w_descent = w_descent2
    i = i+1

end_time_descent = (time.time()) * 1000
time_descent = end_time_descent - start_time_descent
print(time_descent)

RSS_descent = np.matmul(np.transpose(np.subtract(Y,(np.matmul(X_3, w_descent)))),np.subtract(Y,(np.matmul(X_3, w_descent))))
MSE_descent = RSS_descent/10000.000