# -*- coding: utf-8 -*-

'''
COMP551 PROJECT 1

LINEAR REGRESSION

Vincent Bazinet
Gabriel Desrosiers-Grégoire
Kevin Ma

***MAKE SURE THAT ALL THE FILES FOR THE PROJECT ARE IN THE SAME DIRECTORY

'''

import numpy as np
import matplotlib.pyplot as plt
import time

#load pre-processed data (COMMENT OUT THE FOLLOWING IMPORT STATEMENT IF DATA IS ALREADY LOADED)
import preprocessing as pp

'''
INITIALIZATION OF THE DIFFERENT DATA SETS USED:
    
    X_3    -> 3 main features
    X_63   -> 3 main features + 60 most popular words
    X_64_A -> 63-features with (#children) * (length)
    X_64_B -> 63-features with (#children) ^ (is_root + 1)
    X_65   -> 63-features with the extra two features
    X_163  -> 3 main features + 160 most popular words
    X_165  -> 163-features with the extra two features
 
'''

#TRAINING SETS

X_3 = pp.X_train[:,:4]
X_63 = pp.X_train[:,:64]  
X_163 = pp.X_train[:,:164] 

X_64_A = pp.X_train[:,:165]
X_64_A = np.delete(X_64_A, np.arange(64, 164), 1)

X_64_B = pp.X_train[:,:166] 
X_64_B = np.delete(X_64_B, np.arange(64,165),1)

X_165 = pp.X_train[:,:166] # 3 MAIN FEATURES + 160 MOST POPULAR WORDS + BOTH EXTRA FEATURE

X_65 = pp.X_train[:,:166]  # 3 MAIN FEATURES + 60 MOST POPULAR WORDS + BOTH EXTRA FEATURE
X_65 = np.delete(X_65, np.arange(64,164),1)

X_5 = pp.X_train[:,:166]
X_5 = np.delete(X_5, np.arange(4, 164), 1)

#VALIDATION SETS

X_valid_3 = pp.X_valid[:,:4]
X_valid_63 = pp.X_valid[:,:64]
X_valid_163 = pp.X_valid[:,:164]

X_valid_64_A = pp.X_valid[:,:165]
X_valid_64_A = np.delete(X_valid_64_A, np.arange(64, 164), 1)

X_valid_64_B = pp.X_valid[:,:166] 
X_valid_64_B = np.delete(X_valid_64_B, np.arange(64,165),1)

X_valid_165 = pp.X_valid[:,:166]

X_valid_65 = pp.X_valid[:,:166]
X_valid_65 = np.delete(X_valid_65, np.arange(64,164),1)

#TEST SETS

X_test_65 = pp.X_test[:,:166]
X_test_65 = np.delete(X_test_65, np.arange(64,164),1)

Y = pp.Y_train
Y_valid = pp.Y_valid
Y_test = pp.Y_test


'''
LINEAR REGRESSION --- CLOSED-FORM SOLUTION
'''

#functions that computes the Least Squares Estimates
def closed_form(X,Y):    
    return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)

#functions that computes the Mean Square Error (MSE)
def mse(X, Y, w):
    return np.mean((Y-np.matmul(X, w))**2)

'''
###LINEAR REGRESSION --- GRADIENT DESCENT
'''

#gradient descent algorithm
def gradient_descent(X, Y, n_0 = 10**-5, epsilon = 10**-4, eval_MSE=True, B=1, verbose=False, max_iter=None):
    ###Initial weights
#   w_descent = np.random.rand(np.size(X,1))/100
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
PRINTING RESULTS
'''

print("OPTIMIZATION OF THE HYPERPARAMETERS:")
print("NOTE: some of these test might have been commented out")
print("Influence of including the B parameter:")
#Influence of including the B parameter

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

#Gradient-Descent performance across different learning rates
print("Gradient-Descent performance across different learning rates:")

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

print("RUN-TIME COMPARISON:")

#Closed-Form Run-Time (milliseconds)
print("Closed-Form Run-Time (milliseconds)      :")

###compute running time for features == 3
start_time_closed = (time.time()) * 1000  
  
w_closed_3 = closed_form(X_3,Y)

end_time_closed = (time.time()) * 1000
time_closed = end_time_closed - start_time_closed

print(time_closed)

#Gradient-Descent Run-Time (milliseconds)
print("Gradient-Descent Run-Time (milliseconds) :")

start_time_descent_B = (time.time()) * 1000

n_0=1e-4
epsilon=1e-4
B=1/10
[w_descent, MSE_descent, w_dif] = gradient_descent(X_3, Y, n_0 = n_0, epsilon = epsilon, eval_MSE=False, B=B, verbose=False)

end_time_descent_B = (time.time()) * 1000
time_descent_B = end_time_descent_B - start_time_descent_B
print(time_descent_B)

#Evaluating stability
print("STABILITY ANALYSIS:")

#evaluate closed form stability
print("Stability of the closed-form:")

weights=np.zeros([10,X_3.shape[1]])
for i in range(10):
    weights[i,:]=closed_form(X_3[i*1000:(i+1)*1000,:],Y[i*1000:(i+1)*1000])
closed_stability=np.mean(np.std(weights,0))
print(closed_stability)

#Stability of the gradient-descent
print("Stability of the gradient-descent:")

weights=np.zeros([10,X_3.shape[1]])
n_0=1e-4
epsilon=1e-4
B=1/10
for i in range(10):
    [weights[i,:],MSE_descent, w_dif] = gradient_descent(X_3[i*1000:(i+1)*1000,:],Y[i*1000:(i+1)*1000], n_0 = n_0, epsilon = epsilon, eval_MSE=True, B=B, verbose=False)
gradient_stability=np.mean(np.std(weights,0))
print(gradient_stability)

#Gradient descent on model with 3 features

X_descent_train=X_3
X_descent_valid=X_valid_3
n_0=1e-4
epsilon=1e-4
B=1/10
[w_descent, MSE_descent, w_dif] = gradient_descent(X_descent_train, Y, n_0 = n_0, epsilon = epsilon, eval_MSE=True, B=B, verbose=False)
MSE_descent_train=np.mean((Y-np.matmul(X_descent_train, w_descent))**2)
MSE_descent_valid=np.mean((Y_valid-np.matmul(X_descent_valid, w_descent))**2)

#compute coefficients (w) for the different closed-form models
w_closed_63 = closed_form(X_63,Y)
w_closed_163 = closed_form(X_163,Y)
w_closed_64_A = closed_form(X_64_A, Y)
w_closed_64_B = closed_form(X_64_B, Y)
w_closed_165 = closed_form(X_165,Y)
w_closed_65 = closed_form(X_65,Y)

#compute mean squared error (MSE) on training set (closed-form)
MSE_closed_train_3 = mse(X_3, Y, w_closed_3)
MSE_closed_train_63 = mse(X_63, Y, w_closed_63)
MSE_closed_train_163 = mse(X_163, Y, w_closed_163)

MSE_closed_train_64_A = mse(X_64_A, Y, w_closed_64_A)
MSE_closed_train_64_B = mse(X_64_B, Y, w_closed_64_B)

MSE_closed_train_65 = mse(X_65, Y, w_closed_65)
MSE_closed_train_165=np.mean((Y-np.matmul(X_165, w_closed_165))**2)

#compute mean squared error (MSE) on validation set (closed-form)

MSE_closed_val_63 = mse(X_valid_63, Y_valid, w_closed_63)
MSE_closed_val_64_A = mse(X_valid_64_A, Y_valid, w_closed_64_A)
MSE_closed_val_64_B = mse(X_valid_64_B, Y_valid, w_closed_64_B)

MSE_closed_val_3=np.mean((Y_valid-np.matmul(X_valid_3, w_closed_3))**2)
MSE_closed_val_163=np.mean((Y_valid-np.matmul(X_valid_163, w_closed_163))**2)    
MSE_closed_val_165=np.mean((Y_valid-np.matmul(X_valid_165, w_closed_165))**2)
MSE_closed_val_65=np.mean((Y_valid-np.matmul(X_valid_65, w_closed_65))**2)

#compute the gain obtained with our extra 2 features (closed-form)

gain_63 = MSE_closed_val_63 - MSE_closed_val_65
gain_64_A = MSE_closed_val_63 - MSE_closed_val_64_A
gain_64_B = MSE_closed_val_63 - MSE_closed_val_64_B
gain_163 = MSE_closed_val_163 - MSE_closed_val_165

#compute mean squared error(MSE) on test set for our best model
#MSE_closed_test_3=np.mean((Y_test-np.matmul(X_test_3, w_closed_3))**2)
#MSE_closed_test_63=np.mean((Y_test-np.matmul(X_test_63, w_closed_63))**2)
MSE_closed_test_65=np.mean((Y_test-np.matmul(X_test_65, w_closed_65))**2)

print("PERFORMANCE OF OUR MODELS ON THE VALIDATION SET (MSE):")
print("MSE for 3-features model (closed):                               ",MSE_closed_val_3 )
print("MSE for 3-features model (gradient-descent):                     ",MSE_descent_valid)
print("MSE for 63-features model (closed):                              ",MSE_closed_val_63)
print("MSE for 64-features model [(#children) * (length)] (closed):     ",MSE_closed_val_64_A)
print("MSE for 64-features model [(#children) ^ (is_root + 1)] (closed):",MSE_closed_val_64_B)
print("MSE for 65-features model (closed) :                             ",MSE_closed_val_65)
print("MSE for 163-features model (closed):                             ",MSE_closed_val_163)
print("MSE for 165-features model (closed):                             ",MSE_closed_val_165)
print("MSE RESULTS OF OUR MODELS ON THE TEST SET:")
print("MSE FOR 65-features model (closed):                              ",MSE_closed_test_65)
print("MSE GAIN FOR OUR NEW FEATURES:")
print("MSE gain for 165 vs 163 features model:                          ",gain_163)
print("MSE gain for 65 vs 63 features model:                            ",gain_63)
print("MSE gain for [(#children) * (length)]  (vs 63-features):         ",gain_64_A)
print("MSE gain for [(#children) ^ (is_root + 1)] (vs 63-features):     ",gain_64_B)