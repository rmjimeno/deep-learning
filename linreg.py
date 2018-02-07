# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:25:24 2018
@author: JIMENO, Rheanne Odessa M.
"""

#LINEAR REGRESSION THROUGH GRADIENT DESCENT
print("Program may take a while especially for higher order polynomials. Increase number of iterations for more accurate predictions.")    

import sys 
import numpy as np
import matplotlib.pyplot as plt

def linreg(*coeff):  
    x = np.arange(-10,10)
    
    y = 0
    for i in range(0,len(coeff)): 
        y = y + coeff[i]*x**(len(coeff)-(i+1)) + np.random.uniform(0,0.5,len(x))

    
    if(len(coeff)==1):#notes: 0.001 learning rate, 2000 iterations
        b = 1
        learning_rate = 0.001
        iterations = 2000
        sse = 0*np.arange(iterations)
      
        for i in range (0,iterations):
            y_predict = b
            error = y - y_predict
            sse[i] = 1/2 * sum(np.multiply(error,error))    
            grad_b = -error
            b = b - (sum(grad_b))*learning_rate
        y_predict = b*np.ones(len(y))   
        print(b)

    elif(len(coeff)==2):#notes: 0.001 learning rate, 2000 iterations
        a = b = 1
        learning_rate = 0.001
        iterations = 2000
        sse = 0*np.arange(iterations)
      
        for i in range (0,iterations):
            y_predict = a*x + b
            error = y - y_predict
            sse[i] = 1/2 * sum(np.multiply(error,error))    
            grad_a = np.multiply(-x,error)
            grad_b = -error
            a = a - (sum(grad_a))*learning_rate
            b = b - (sum(grad_b))*learning_rate
        y_predict = a*x + b
        print(a,b)


    elif(len(coeff)==3):#notes: 0.001 learning rate, 200k iterations
        a = b = c = 1
        learning_rate = 0.00001
        iterations = 200000
        sse = 0*np.arange(iterations)
      
        for i in range (0,iterations):
            y_predict = c*x**2 + a*x + b
            error = y - y_predict
            #sse[i] = 1/2 * sum(np.multiply(error,error))
            sse[i] = sum(error)/len(x)   
            grad_c = np.multiply(-x**2,error)
            grad_a = np.multiply(-x,error)
            grad_b = -error
            c = c - (sum(grad_c))*learning_rate
            a = a - (sum(grad_a))*learning_rate
            b = b - (sum(grad_b))*learning_rate
        y_predict = c*x**2 + a*x + b   
        print(c,a,b)

    elif(len(coeff)==4):#notes: 0.00001 learning rate, 750k iterations
        a = b = c = d = 1
        learning_rate = 0.00001
        iterations = 750000
        sse = 0*np.arange(iterations)
      
        for i in range (0,iterations):
            y_predict = d*x**3 + c*x**2 + a*x + b
            error = y - y_predict
            #sse[i] = 1/2 * sum(np.multiply(error,error))
            sse[i] = sum(error)/len(x)
            
            grad_d = np.multiply(-x**3,error)/len(x)
            grad_c = np.multiply(-x**2,error)/len(x)
            grad_a = np.multiply(-x,error)/len(x)
            grad_b = -error/(len(x))
            
            d = d - (sum(grad_d))*learning_rate        
            c = c - (sum(grad_c))*learning_rate
            a = a - (sum(grad_a))*learning_rate
            b = b - (sum(grad_b))*learning_rate
        y_predict = d*x**3 + c*x**2 + a*x + b
        print(d,c,a,b)

    else:
        print("Program accept up to 3rd degree polynomial only.")
    
    plt.plot(x,y_predict,'r-')
    plt.plot(x,y,'o')
    plt.show()

#terminal
args = 0*np.arange(len(sys.argv)-1)
for i in range (1,len(sys.argv)):
    args[i-1] = sys.argv[i]

if (len(sys.argv)==2):
    arg0 = args[0]
    linreg(arg0) 
elif (len(sys.argv)==3):
    arg0 = args[0]    
    arg1 = args[1]
    linreg(arg0, arg1)  
elif (len(sys.argv)==4): 
    arg0 = args[0]    
    arg1 = args[1]
    arg2 = args[2]
    linreg(arg0, arg1, arg2)  
elif (len(sys.argv)==5):
    arg0 = args[0]    
    arg1 = args[1]
    arg2 = args[2]    
    arg3 = args[3]
    linreg(arg0, arg1, arg2, arg3)  
else:
    print("Error: Program can take up to 3rd degree polynomial only.")
  
