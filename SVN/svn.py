from cvxopt.solvers import qp
from cvxopt . base import matrix
import numpy as np
import pandas as pa
import pylab
import random
import math

# Generate the random dataset
def random_data():

# It will create 10 data points for each class
    np.random.seed(100)
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
    random. shuffle (data)

    #
    pylab.hold(True)
    pylab.plot([p[0] for p in classA],[p[1] for p in classA],'bo')
    pylab.plot([p[0] for p in classB],[p[1] for p in classB],'ro')
    pylab .show()

# Linear kernal function

def main():

    random_data()

if __name__ == "__main__":
    main()
