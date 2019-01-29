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
import matplotlib.pyplot as plt
import time

#load pre-processed data (COMMENT OUT THE FOLLOWING IMPORT STATEMENT IF DATA IS ALREADY LOADED)
import preprocessing as pp

 #Initialize the data set 
X_3 = pp.X_train[:,:4]
X_63 = pp.X_train[:,:64]
X_163 = pp.X_train[:,:164]
X_164 = pp.X_train[:,:165]
X_165 = pp.X_train[:,:166]

X_64 = pp.X_train[:,:166]
X_64 = np.delete(X_64, np.arange(64,165),1)

X_65 = pp.X_train[:,:166]
X_65 = np.delete(X_65, np.arange(64,164),1)

X_valid_3 = pp.X_valid[:,:4]
X_valid_63 = pp.X_valid[:,:64]
X_valid_163 = pp.X_valid[:,:164]
X_valid_164 = pp.X_valid[:,:165]
X_valid_165 = pp.X_valid[:,:166]

X_valid_64 = pp.X_valid[:,:166]
X_valid_64 = np.delete(X_valid_64, np.arange(64,165),1)

X_valid_65 = pp.X_valid[:,:166]
X_valid_65 = np.delete(X_valid_65, np.arange(64,164),1)

Y = pp.Y_train
Y_valid = pp.Y_valid

###LINEAR REGRESSION --- CLOSED-FORM SOLUTION

def closed_form(X,Y):    
    return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)

#compute coefficients (w)

w_closed_63 = closed_form(X_63,Y)
w_closed_163 = closed_form(X_163,Y)
w_closed_164 = closed_form(X_164,Y)
w_closed_165 = closed_form(X_165,Y)
w_closed_65 = closed_form(X_65,Y)
w_closed_64 = closed_form(X_64,Y)

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
MSE_closed_train_164=np.mean((Y-np.matmul(X_164, w_closed_164))**2)
MSE_closed_train_165=np.mean((Y-np.matmul(X_165, w_closed_165))**2)
MSE_closed_train_65=np.mean((Y-np.matmul(X_65, w_closed_65))**2)
MSE_closed_train_64=np.mean((Y-np.matmul(X_64, w_closed_64))**2)

#compute mean squared error (MSE) on validation set
MSE_closed_val_3=np.mean((Y_valid-np.matmul(X_valid_3, w_closed_3))**2)
MSE_closed_val_63=np.mean((Y_valid-np.matmul(X_valid_63, w_closed_63))**2)
MSE_closed_val_163=np.mean((Y_valid-np.matmul(X_valid_163, w_closed_163))**2)    
MSE_closed_val_164=np.mean((Y_valid-np.matmul(X_valid_164, w_closed_164))**2)
MSE_closed_val_165=np.mean((Y_valid-np.matmul(X_valid_165, w_closed_165))**2)
MSE_closed_val_65=np.mean((Y_valid-np.matmul(X_valid_65, w_closed_65))**2)
MSE_closed_val_64=np.mean((Y_valid-np.matmul(X_valid_64, w_closed_64))**2)


###LINEAR REGRESSION --- GRADIENT DESCENT
def gradient_descent(X, Y, n_0 = 10**-5, epsilon = 10**-4, eval_MSE=True, B=1, verbose=False, max_iter=None):
    ###Initial weights
#    w_descent = np.random.rand(np.size(X,1))/100
    w_descent = np.zeros(np.size(X,1))

    #keep track of the error along the gradient descent
    MSE_descent=[]
    w_dif=[]
    i = 1    
    alpha = n_0
    while True:    
        if eval_MSE:
            #EVAL THE MSE for the current coefficients 
            MSE=np.mean((Y-np.matmul(X, w_descent))**2)
            MSE_descent.append(MSE)
            if verbose:
                print(MSE)
                    
        #compute the new coefficients
        w_descent2=w_descent-(2*alpha*(np.matmul(np.transpose(X).dot(X),w_descent)-np.transpose(X).dot(Y)))
        dif=np.linalg.norm(w_descent - w_descent2)
        w_dif.append(dif)
        
        #stop when descent is completed
        if dif < epsilon:
            break
        w_descent = w_descent2
        alpha = n_0/(B*i+1)
        i = i+1
        if not max_iter==None and max_iter<i:
            break
        
    #(COMMENT OUT LAST 2 OUTPUTS IF YOU WANT THE RUNNING TIME)
    return w_descent, np.asarray(MSE_descent), w_dif

'''
Influence of including the B parameter
'''
X=X_3
n_0=1e-5
epsilon=1e-4
max_iter=100
fig,[ax1,ax2]=plt.subplots(2,1)
B=1
[w_descent, MSE_descent, w_dif] = gradient_descent(X, Y, n_0 = n_0, epsilon = epsilon,B=B, verbose=True,max_iter=max_iter)
ax1.plot(MSE_descent)
ax2.plot(w_dif)
B=1/10
[w_descent, MSE_descent, w_dif] = gradient_descent(X, Y, n_0 = n_0, epsilon = epsilon,B=B, verbose=True,max_iter=max_iter)
ax1.plot(MSE_descent)
ax2.plot(w_dif)
ax2.set_yscale("log")
B=0
#n_0=1e-6
[w_descent, MSE_descent, w_dif] = gradient_descent(X, Y, n_0 = n_0, epsilon = epsilon,B=B,verbose=True,max_iter=max_iter)
ax1.plot(MSE_descent)
ax2.plot(w_dif)
#ax2.legend(["B=1; alpha=1e-5","B=1/10; alpha=1e-5","B=0; alpha=1e-6"])
ax2.legend(["B=1","B=1/10","B=0"])
ax2.set_xlabel("Gradient Descent Iteration", fontsize=12)
ax1.set_ylabel("Mean Square Error", fontsize=12)
ax2.set_ylabel("L2 norm of weights", fontsize=12)
    

'''
Gradient-Descent performance across different learning rates
'''
X=X_63
X_valid=X_valid_63
num_iter=10
n_0=1e-5
epsilon=1e-4
B=1/10
learning_rates=np.zeros(num_iter)
errors=np.zeros(num_iter)
valid_errors=np.zeros(num_iter)
descent_iterations=np.zeros(num_iter)
for i in range(num_iter):
    epsilon=n_0*10
    [w_descent, MSE_descent, w_dif] = gradient_descent(X, Y, n_0 = n_0, epsilon = epsilon, eval_MSE=True, B=B, verbose=True)
    errors[i]=MSE_descent[-1]
    descent_iterations[i]=len(MSE_descent)
    valid_errors[i]=np.mean((Y_valid-np.matmul(X_valid, w_descent))**2)
    learning_rates[i]=n_0
    n_0=n_0-n_0/10
    print(i)

fig,ax1=plt.subplots(1,1)
ax1.plot(errors, learning_rates)
ax1.plot(valid_errors, learning_rates)
ax1.set_yscale("log")
ax1.set_xlabel("Final mean square error", fontsize=15)
ax1.set_ylabel("Learning rate", fontsize=15)
ax1.legend(['Training set','Validation set'])

fig,ax1=plt.subplots(1,1)
ax1.plot(descent_iterations, learning_rates)
ax1.set_yscale("log")
ax1.set_xlabel("Number of iterations", fontsize=15)
ax1.set_ylabel("Learning rate", fontsize=15)

'''
Gradient-Descent Run-Time
'''

start_time_descent_B = (time.time()) * 1000

n_0=1e-4
epsilon=1e-4
B=1/10
[w_descent, MSE_descent, w_dif] = gradient_descent(X_3, Y, n_0 = n_0, epsilon = epsilon, eval_MSE=False, B=B, verbose=False)

end_time_descent_B = (time.time()) * 1000
time_descent_B = end_time_descent_B - start_time_descent_B
print(time_descent_B)


'''
Evaluating stability
'''

#evaluate closed form stability
weights=np.zeros([10,X_3.shape[1]])
for i in range(10):
    weights[i,:]=closed_form(X_3[i*1000:(i+1)*1000,:],Y[i*1000:(i+1)*1000])
closed_stability=np.mean(np.std(weights,0))

weights=np.zeros([10,X_3.shape[1]])
n_0=1e-4
epsilon=1e-4
B=1/10
for i in range(10):
    [weights[i,:],MSE_descent, w_dif] = gradient_descent(X_3[i*1000:(i+1)*1000,:],Y[i*1000:(i+1)*1000], n_0 = n_0, epsilon = epsilon, eval_MSE=True, B=B, verbose=False)
gradient_stability=np.mean(np.std(weights,0))

'''
Gradient descent performance on 3 features
'''
X=X_163
X_valid=X_valid_163
n_0=1e-5
epsilon=1e-4
B=1/10
[w_descent, MSE_descent, w_dif] = gradient_descent(X, Y, n_0 = n_0, epsilon = epsilon, eval_MSE=True, B=B, verbose=True)
MSE_train=np.mean((Y-np.matmul(X, w_descent))**2)
MSE_valid=np.mean((Y_valid-np.matmul(X_valid, w_descent))**2)

