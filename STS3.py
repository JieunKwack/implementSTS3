import pandas as pd
import datetime as dt
import numpy as np
from preprocessing import Bound, readDataasDF, TimeSeriesTranstoSet, QueryTranstoSet, Bound_q, NN, Trans_outQuery_to_Set, Jaccard
import csv

# set parameter cell size
epsilon = 76
sigma = 0.82

# read some database
D = readDataasDF('CinC_ECG_torso\\CinC_ECG_torso_TRAIN')
Q = readDataasDF('CinC_ECG_torso\\CinC_ECG_torso_TEST')

BD = Bound(D) # compute Bound
D_trans = TimeSeriesTranstoSet(D, BD, epsilon, sigma)

ans = NN()
errorRate = 0
for q in Q.index:
    query = Q.loc[q]
    BQ = Bound_q(query)
    if ((BQ.xmin < BD.xmin) | (BD.xmax < BQ.xmax)):# check Query bound out of the DB bound
        Q_trans = Trans_outQuery_to_Set(query, BD, epsilon, sigma)
    else:
        Q_trans = QueryTranstoSet(query, BD, epsilon, sigma)
    for i in D_trans.index:
        Q_trans = set(filter(lambda x: x == x , Q_trans))
        Target = set(filter(lambda x: x == x , D_trans.loc[i]))
        jac = Jaccard(set(Target), set(Q_trans))
        # print(set(D_trans.loc[i]))
        # print(Q_trans)
        # with open("result.csv", "a") as file:
        #     csv.writer(file).writerow(jac)
        print("jac: ",jac)
        if (ans.Jac < jac):
            ans.TS = D_trans.loc[i]
            ans.Jac = jac
            ans.label = D.label[i]
    if (query.label != ans.label):
        errorRate = errorRate + 1
    # print(q)

print("errorRate: ", errorRate / len(Q.index))
