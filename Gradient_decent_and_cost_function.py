import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#y = mx + b

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iteration = 10
    n = len(x)
    learning_rate = 0.01
    #plt.scatter(x,y,color="red",marker="+",linewidth= '5')
    #plt.show()
    for i in range(iteration):
        y_predicted = m_curr*x + b_curr
        

        cost =(1/n)*sum([val**2 for val in (y-y_predicted)])
        
        #plt.plot(x,y_predicted,color='green')
        
        m_derivative = -(2/n) * sum(x*(y-y_predicted))
        b_derivative = -(2/n) * sum(y-y_predicted)
        
        m_curr = m_curr - learning_rate * m_derivative
        b_curr = b_curr - learning_rate * b_derivative
        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost,i))
        


x = np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x,y)

