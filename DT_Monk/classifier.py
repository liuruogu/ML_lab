import numpy as np
import dtree as d
import pandas as pd
import monkdata as m
import matplotlib.pyplot as plt
import random
import sys

#calculate the entropy of the dataset 
def CalEntropy():
    e1 = d.entropy(m.monk1);
    e2 = d.entropy(m.monk2);
    e3 = d.entropy(m.monk3);
    print("The entropy of three data set: ",e1,e2,e3)

def CalGini():
#calculate the info gain to six attributes in monk datasets
    for i in range(len(m.attributes)):
        gi = d.averageGain(m.monk1,m.attributes[i])
        print("Info gain of the no.",i+1,"attribute in monk1:", gi)
    print("-----------------------------------------------------")

    for i in range(len(m.attributes)):
        gi = d.averageGain(m.monk2,m.attributes[i])
        print("Info gain of the no.",i+1,"attribute in monk2:", gi)    
    print("-----------------------------------------------------")

    for i in range(len(m.attributes)):
        gi = d.averageGain(m.monk3,m.attributes[i])
        print("Info gain of the no.",i+1,"attribute in monk3:", gi)
    print("-----------------------------------------------------")

def subEntropy():
    sub = d.select(m.monk1,m.attributes[4],1)
    sub2 = d.select(m.monk1,m.attributes[4],2)
    sub3 = d.select(m.monk1,m.attributes[4],3)
    sub4 = d.select(m.monk1,m.attributes[4],4)
    # The combined of 
    sub1 = sub2+sub3+sub4
    sube = d.entropy(sub1)
    # print(len(sub1)
    print("The sub entropy of monk1 that attribute 5 = 1 is", sube)

def buildT():
#build the decision tree for the monk1 dataset and test its error rate using testing set
    t = d.buildTree(m.monk3, m.attributes)
    print(t)
    print(d.check(t, m.monk3test))

# add the prunning process 
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def maxRate (treeList,train,val):
    max1 = 0
    t = treeList[0]
    for t in treeList: 
        if max1 < d.check(t,val):
            max1 = d.check(t,val)  
            # print(max1)
            maxTree = t
            # print(maxTree)
    return max1, maxTree

def prunTree(data):
    # All possible partition
    par = [0.3,0.4,0.5,0.6,0.7,0.8]
    # Used to accumulate the correct rate result
    correctRate = [0,0,0,0,0,0]
    # newRate = []
    # Try every partition and find the best one 
    # j = 0
    result=pd.DataFrame(index=range(0,1000),columns=par)

    for t in range(1000):
        i = 0 
        for p in par:
            monktrain, monkval = partition(data, p)
            TestTr = d.buildTree(monktrain,m.attributes)
            # Get all possibily prunned trees
            pt = d.allPruned(TestTr)
            # print("******************************Partition rate", p)
            # Go through all the prunned trees
            maxR,maxT = maxRate(pt,monktrain,monkval)
            pt2 = d.allPruned(maxT)
            maxR2,maxT2 = maxRate(pt2,monktrain,monkval)
            while maxR < maxR2:
                maxR3, maxT3 = maxRate(d.allPruned(maxT2),monktrain,monkval)
                if maxR2 < maxR3:
                    maxR = maxR3
                    maxT2 = maxT3
                else:
                    maxR = maxR2 
                    maxT = maxT2
                    break
            correctRate[i] = correctRate[i]+maxR
            i = i+1
            # j = j+1
            print(maxR) 
            print(t)
            print(p)
            result.set_value(t, p, 1-maxR)
    print(result)
    result.plot(kind = 'kde', title = '100 times of different fraction' ,legend=True)
    plt.ylabel('Density')
    plt.xlabel('Error')
    plt.show()

    # newRate = [1-i/1000 for i in correctRate]
    # plt.title('Error rates(mean) in all partitions')
    # print(newRate)
    
    # x = [0.3,0.4,0.5,0.6,0.7,0.8]
    # y = newRate
    # plt.xlim((0,1))
    # plt.plot(x, y,'ro')
    # plt.xlabel('Partitions')
    # plt.ylabel('Error rates')
    # plt.show()


# Get the spread meature for the data

def main(argv):
    # times = [100, 1000]
#    CalEntropy() 
#    CalGini()
#    subEntropy()
#    buildT()
    prunTree(m.monk3)
    
if __name__ == "__main__":
    main(sys.argv)