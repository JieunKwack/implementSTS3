import pandas as pd
import datetime as dt
import numpy as np
from preprocessing import Bound, readDataasDF, TimeSeriesTranstoSet, QueryTranstoSet, Bound_q, NN, Trans_outQuery_to_Set, Jaccard
import csv

# set parameter cell size
epsilon = 21
sigma = 0.18

# read some database
D = readDataasDF('CBF\\CBF_TRAIN')
Q = readDataasDF('CBF\\CBF_TEST')

BD = Bound(D) # compute Bound
# print(BD.tmin, BD.tmax, BD.xmin, BD.xmax)
D_trans = TimeSeriesTranstoSet(D, BD, epsilon, sigma)

errorRate = 0
for q in Q.index:
    ans = NN()
    query = Q.loc[q]#Q.loc[q]
    BQ = Bound_q(query)
    if ((BQ.xmin < BD.xmin) | (BD.xmax < BQ.xmax)):# check Query bound out of the DB bound
        Q_trans = Trans_outQuery_to_Set(query, BD, epsilon, sigma)
    else:
        Q_trans = QueryTranstoSet(query, BD, epsilon, sigma)
    for i in D_trans.index:
        Q_trans = set(filter(lambda x: x == x , Q_trans)) # remove NAN
        Target = set(filter(lambda x: x == x , D_trans.loc[i]))
        jac = Jaccard(set(Target), set(Q_trans))
        # print(set(D_trans.loc[i]))
        # print(Q_trans)
        # with open("result.csv", "a") as file:
        #     csv.writer(file).writerow(jac)
        # print("jac: ",jac)
        if (ans.Jac < jac):
            ans.TS = i # index: D_trans.loc[i]
            ans.Jac = jac
            ans.label = D.label[i]
    if (query.label != ans.label):
        errorRate = errorRate + 1
        # print(ans.TS)
        # print("wrong:", ans.TS, query)
        # print("index:", q)

print("errorRate: ", errorRate / len(Q.index))
# print(errorRate)
