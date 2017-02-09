import pandas as pd
import datetime as dt
import numpy as np
from preprocessing import Bound, readDataasDF, TimeSeriesTranstoSet, QueryTranstoSet, Bound_q, NN, Trans_outQuery_to_Set, Jaccard
import csv

# set parameter cell size
sigma = 0.88
epsilon = 4

# read some database
D = readDataasDF('ElectricDevices\\ElectricDevices_TRAIN')
Q = readDataasDF('ElectricDevices\\ElectricDevices_TEST')

BD = Bound(D) # compute Bound
print(BD.xmin, BD.xmax)
D_trans = TimeSeriesTranstoSet(D, BD, epsilon, sigma) #
print("done transtoSet")
errorRate = 0
for q in Q.index:
    query = Q.loc[q]
    ans = NN()
    BQ = Bound_q(query)
    if ((BQ.xmin < BD.xmin) | (BD.xmax < BQ.xmax)):# check Query bound out of the DB bound
        Q_trans = Trans_outQuery_to_Set(query, BD, epsilon, sigma)
    else:
        Q_trans = QueryTranstoSet(query, BD, epsilon, sigma)
    for i in D_trans.index:
        Q_trans = set(filter(lambda x: x == x , Q_trans)) # remove NAN
        # print(Q_trans)
        Target = set(filter(lambda x: x == x , D_trans.loc[i]))
        # print(Target)
        jac = Jaccard(set(Target), set(Q_trans))
        if (ans.Jac < jac):
            ans.TS = i # index: D_trans.loc[i]
            ans.Jac = jac
            ans.label = D.label[i]
    if (query.label != ans.label):
        errorRate = errorRate + 1
        # print("index q&i: ", q, ans.TS)
        # print("jac: ", ans.Jac)

print("errorRate: ", errorRate / len(Q.index))
# print(errorRate)
