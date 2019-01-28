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
X_3 = pp.X_train[:,:4]
X_63 = pp.X_train[:,:64]
X_163 = pp.X_train[:,:164]

X_valid_3 = pp.X_valid[:,:4]
X_valid_63 = pp.X_valid[:,:64]
X_valid_163 = pp.X_valid[:,:164]

Y = pp.Y_train
Y_valid = pp.Y_valid

###LINEAR REGRESSION --- CLOSED-FORM SOLUTION

def closed_form(X,Y):    
    return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)

#compute coefficients (w)

w_closed_63 = closed_form(X_63,Y)
w_closed_163 = closed_form(X_163,Y)

###compute running time for features == 3
start_time_closed = (time.time()) * 1000  
  
w_closed_3 = closed_form(X_3,Y)

end_time_closed = (time.time()) * 1000
time_closed = end_time_closed - start_time_closed
print(time_closed)

#compute mean squared error (MSE) on training set
MSE_closed_train_3=np.mean((Y-np.matmul(X_3, w_closed_3))**2)
MSE_closed_train_63=np.mean((Y-np.matmul(X_63, w_closed_63))**2)
MSE_closed_train_163=np.mean((Y-np.matmul(X_163, w_closed_163))**2)

#compute mean squared error (MSE) on validation set
MSE_closed_val_3=np.mean((Y_valid-np.matmul(X_valid_3, w_closed_3))**2)
MSE_closed_val_63=np.mean((Y_valid-np.matmul(X_valid_63, w_closed_63))**2)
MSE_closed_val_163=np.mean((Y_valid-np.matmul(X_valid_163, w_closed_163))**2)    

###LINEAR REGRESSION --- GRADIENT DESCENT
def gradient_descent(X, Y, n_0 = 10**-5, epsilon = 10**-4, use_B=True):
    ###Initial weights
    w_descent = np.random.rand(np.size(X,1))

    #keep track of the error along the gradient descent
    MSE_descent=[]
    w_dif=[]
    i = 1    
    B=0
    while True:    
        #PRINT THE MSE for the current coefficients 
        #(COMMENT OUT IF YOU WANT THE RUNNING TIME)
        MSE=np.mean((Y-np.matmul(X, w_descent))**2)
        print(MSE)
        MSE_descent.append(MSE)
        
        alpha = n_0/((B+i)+1)
        
        #compute the new coefficients
        w_descent2=w_descent-(2*alpha*(np.matmul(np.transpose(X).dot(X),w_descent)-np.transpose(X).dot(Y)))
        dif=np.linalg.norm(w_descent - w_descent2)
       
        #(COMMENT OUR IF YOU WANT THE RUNNING TIME)
        w_dif.append(dif)
        
        #stop when descent is completed
        if dif < epsilon:
            break
        w_descent = w_descent2
        i = i+1

        if use_B:
            B += 1
        
    #(COMMENT OUT LAST 2 OUTPUTS IF YOU WANT THE RUNNING TIME)
    return w_descent, np.asarray(MSE_descent), w_dif

'''
Gradient-Descent Run-Time
'''

'''
start_time_descent_B = (time.time()) * 1000

w_descent_B = gradient_descent(X_3, Y, n_0 = 10**-5, epsilon = 10**-4,use_B=True)

end_time_descent_B = (time.time()) * 1000
time_descent_B = end_time_descent_B - start_time_descent_B
print(time_descent_B)


start_time_descent_noB = (time.time()) * 1000

w_descent_noB = gradient_descent(X_3, Y, n_0 = 10**-5, epsilon = 10**-4,use_B=False)

end_time_descent_noB = (time.time()) * 1000
time_descent_noB = end_time_descent_noB - start_time_descent_noB
print(time_descent_noB)
'''

'''
Visualization
'''

import matplotlib.pyplot as plt
fig,[ax1,ax2]=plt.subplots(2,1)
[w_descent, MSE_descent, w_dif] = gradient_descent(X_3, Y, n_0 = 10**-5, epsilon = 10**-4,use_B=True)
ax1.plot(MSE_descent)
ax2.plot(w_dif)
ax2.set_yscale("log")
[w_descent, MSE_descent, w_dif] = gradient_descent(X_3, Y, n_0 = 10**-5, epsilon = 10**-4,use_B=False)
ax1.plot(MSE_descent)
ax2.plot(w_dif)
ax2.legend(["Including B","Without B"])


