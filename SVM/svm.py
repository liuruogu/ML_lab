import numpy as np
import pylab
import random
import math
from cvxopt.solvers import qp
from cvxopt.base import matrix


# def random_data():
#   It will create 10 data points for each class
#   np.random.seed(100)
#     P = np.zeros((20, 20))
#     classA = [(random.normalvariate(1.5, 1),
#                 random.normalvariate(0.5, 1),
#                1.0)
#             for i in range(5)] + \
#             [(random.normalvariate(1.5, 1),
#               random.normalvariate(0.5, 1),
#             1.0)
#             for i in range(5)]
#
#     classB = [(random.normalvariate(0.0, 0.5),
#                random. normalvariate(0.5, 0.5),
#              1.0)
#             for i in range(10)]
#
#     data = classA + classB
#     random.shuffle(data)
#
#     # For testing
#     classA1 = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + \
#     [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]
#     classB1 = [(random.normalvariate(3, 0.5), random.normalvariate(-1.5,0.5), -1.0) for i in range(10)]
#     data1 = classA1 + classB1
#     random.shuffle(data1)

    
    # for i in range(20):
    #     for j in range(20):
    #         P[i][j] = poly_kernal(data[i],data[j])


    # Create the vector q and h, and the matrix G
    # q = np.zeros((10,1))
    # for i in range(10):
    #     q[i][0] = -1
    # print(P)
    # h = matrix(0.0, (20,1))
    # q = matrix(-1.0, (20,1))
    # G = np.zeros((20,20))
    # for x in range(20):
    #     for y in range(20):
    #         if x == y:
    #             G[x][y] = -1
    # # print(P,q,G,h)
    #
    # r = solvers.qp(matrix(P), q, matrix(G), h)
    # # print(matrix(P), q, matrix(G), h)
    # alpha = list(r['x'])
    # print("The alpha we get",alpha)
    #
    # # Select the vector for support vector
    # indexes = []
    # for index in range(len(alpha)):
    #     if alpha[index] > 1.0e-5:
    #         indexes.append(index)
    # print("Our data points",data)
    # print("The indexes we get",indexes)
    #
    #
    # for i in range(len(indexes)):
    #     print("The data corresponding with the data",data[indexes[i]])
    #
    # xrange=np.arange( -4, 4, 0.05)
    # yrange=np.arange( -4, 4, 0.05)
    # grid=matrix([[indicator(x, y, alpha,indexes, data) for y in yrange ] for x in xrange])
    # pylab.contour(xrange, yrange, grid, (-0.1, 0.0, 0.1),colors = ('red','black','blue'), linewidths = (1,3,1))
    #
    # pylab.show()

def ori_plot(classA,classB):
    pylab.plot([p[0] for p in classA],[p[1] for p in classA],'bo')
    pylab.plot([p[0] for p in classB],[p[1] for p in classB],'ro')
    pylab.title('Plot data')
    pylab.xlabel('X')
    pylab.ylabel('Y')
    pylab.show()

def linear_kernal(x,y,p):
    return np.dot(x,y)

def poly_kernal(x,y,p):
    return math.pow(np.dot(x,y),p)

def gauss_kernal(x,y,p):
    sigma = p
    return math.exp(-np.dot(np.matrix(x)-np.matrix(y), (np.matrix(x)- np.matrix(y)).transpose())/(2*pow(sigma, 2)))


def get_P(data, K, p):
    P = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            P[i,j] = data[i][2]*data[j][2]*Kernel(data[i][0:2], data[j][0:2], K, p)
    return P

# data training
def train_data(data,Option,p):
    # Create the vector q and h, and the matrix G, and get the optimization result
    P = get_P(data,Option,p)
    h = np.zeros((len(data), 1))
    G = np.diag([-1.0]*len(data))
    q = -1*np.ones((len(data), 1))

    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])
    print(alpha)

    # Set the threshold select the support vector
    support = []
    for i in range(len(alpha)):
        if alpha[i] > 10e-5:
            support.append((data[i][0], data[i][1], data[i][2], alpha[i]))
    return support

# Kernel function
def Kernel(x, y, Option, p):
    kValue = 0
    if Option == "L": # Linear/without kernal
        kValue = np.dot(x, y) + 1
    elif Option == "P": # Polynomal kernal
        kValue = pow(np.dot(x, y) + 1, p)
    elif Option == "R": # Gaussian kernal/Radical Basis Function kernels
        sigma = p
        kValue = math.exp(-np.dot(np.matrix(x)-np.matrix(y), (np.matrix(x)- np.matrix(y)).transpose())/(2*pow(sigma, 2)))

    return kValue

def indicator(xstar, support, Option, p):
    ind = 0
    for i in range(len(support)):
        ind += support[i][3]*support[i][2]*Kernel(xstar, support[i][0:2], Option, p)
    return ind


def boundary_plot(classA, classB, support, Option, p):
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)
    grid = matrix([[indicator([x, y], support, Option, p) for y in yrange] for x in
xrange])
    # plot the decision boundary
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black','blue'), linewidths = (1, 3, 1))
    pylab.plot([p[0] for p in classA],[p[1] for p in classA],'bo')
    pylab.plot([p[0] for p in classB],[p[1] for p in classB],'ro')
    pylab.title('Decision Boundary' )

    pylab.xlabel('X')
    pylab.ylabel('Y')
    pylab.show()
# def indicator(x,y,alpha,indexes,data ):
#     ind = 0;
#     for i in range(len(indexes)):
#         if indexes[i]>9:
#             ind = ind+alpha[indexes[i]]*(-1)*poly_kernal(data[i],(x,y,1))
#         else:
#             ind = ind+alpha[indexes[i]]*(1)*poly_kernal(data[i],(x,y,1))
#         # print(ind)
#     return ind


def main():
    # P = np.zeros((20, 20))
    # classA = [(random.normalvariate(1.5, 1),
    #             random.normalvariate(0.5, 1),
    #            1.0)
    #         for i in range(5)] + \
    #         [(random.normalvariate(1.5, 1),
    #           random.normalvariate(0.5, 1),
    #         1.0)
    #         for i in range(5)]
    #
    # classB = [(random.normalvariate(0.0, 0.5),
    #            random. normalvariate(0.5, 0.5),
    #          1.0)
    #         for i in range(10)]
    #
    # data = classA + classB
    # random.shuffle(data)

    # The original data set
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + \
    [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5,0.5), -1.0) for i in range(10)]
    data = classA + classB
    random.shuffle(data)

    #  New data set for testing
    classA1 = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + \
    [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]
    classB1 = [(random.normalvariate(3, 0.5), random.normalvariate(-1.5,0.5), -1.0) for i in range(10)]
    data1 = classA1 + classB1
    random.shuffle(data1)

    # random_data()
    ori_plot(classA, classB)
    support = train_data(data,'P',1)
    boundary_plot(classA,classB,support,'P',1)


if __name__ == "__main__":
    main()
