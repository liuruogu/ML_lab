import numpy as np
import dtree as d
import monkdata as m

#calculate the entropy of the dataset 
e1 = d.entropy(m.monk1);
e2 = d.entropy(m.monk2);
e3 = d.entropy(m.monk3);
print("The entropy of three data set: ",e1,e2,e3)

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

# 