import numpy as np
import pandas as pa
import pylab
import random
import math
# from cvxopt.solvers import qp
# from cvxopt.base import matrix
from cvxopt import matrix, solvers
#
# Generate the random dataset
def random_data():

# It will create 10 data points for each class
#     np.random.seed(100)
    P = np.zeros((20,20))
    classA = [(random.normalvariate( 1.5, 1),
               random.normalvariate(0.5, 1),
               1.0)
            for i in range(5)] + \
            [(random.normalvariate(1.5, 1),
              random.normalvariate(0.5, 1),
            1.0)
            for i in range(5)]

    classB = [(random.normalvariate(0.0, 0.5),
               random. normalvariate( 0.5, 0.5) ,
             1.0)
            for i in range(10)]

    data = classA + classB

    # print(classA, classB)
    # print(data)

    #Plot two class using blue and red dots
    pylab.hold(True)
    pylab.plot([p[0] for p in classA],[p[1] for p in classA],'bo')
    pylab.plot([p[0] for p in classB],[p[1] for p in classB],'ro')


    for i in range(20):
        # Build the P matrix
        # print(data[i][0],data[i][1])
        # print(data[i])
        for j in range(20):

            # Polynomial kernel
            P[i][j] = poly_kernal(data[i],data[j])


    # Create the vector q and h, and the matrix G
    # q = np.zeros((10,1))
    # for i in range(10):
    #     q[i][0] = -1
    print(P)
    h = matrix(0.0, (20,1))
    q = matrix(-1.0, (20,1))
    G = np.zeros((20,20))
    for x in range(20):
        for y in range(20):
            if x == y:
                G[x][y] = -1
    # print(P,q,G,h)

    r = solvers.qp(matrix(P), q, matrix(G), h)
    # print(matrix(P), q, matrix(G), h)
    alpha = list(r['x'])
    print("The alpha we get",alpha)

    # Select the vector for support vector
    indexes = []
    for index in range(len(alpha)):
        if alpha[index] > 1.0e-5:
            indexes.append(index)
    print("Our data points",data)
    print("The indexes we get",indexes)


    for i in range(len(indexes)):
        print("The data corresponding with the data",data[indexes[i]])

    xrange=np.arange( -4, 4, 0.05)
    yrange=np.arange( -4, 4, 0.05)
    grid=matrix([[indicator(x, y, alpha,indexes, data) for y in yrange ] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-0.1, 0.0, 0.1),colors = ('red','black','blue'), linewidths = (1,3,1))

    pylab.show()

def linear_kernal(x,y):
    # Cal the scalar product of each points
    return np.dot(x,y)

def poly_kernal(x,y):
    return math.pow(np.dot(x,y),1)

def indicator(x,y,alpha,indexes,data ):
    ind = 0;
    for i in range(len(indexes)):
        if indexes[i]>9:
            ind = ind+alpha[indexes[i]]*(-1)*poly_kernal(data[i],(x,y,1))
        else:
            ind = ind+alpha[indexes[i]]*(1)*poly_kernal(data[i],(x,y,1))
        # print(ind)
    return ind

def main():

    random_data()
if __name__ == "__main__":
    main()