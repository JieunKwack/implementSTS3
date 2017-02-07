import pandas as pd
import datetime as dt
import numpy as np
from preprocessing import Bound, readDataasDF, TimeSeriesTranstoSet, QueryTranstoSet, Bound_q, NN, Trans_outQuery_to_Set, Jaccard

# set parameter cell size
epsilon = 21
sigma = 0.18

# read some database
D = readDataasDF('CBF\\CBF_TRAIN')
Q = readDataasDF('CBF\\CBF_TEST')

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
        jac = Jaccard(D_trans.loc[i], Q_trans)
        print(jac)
        if (ans.Jac < jac):
            ans.TS = D_trans.loc[i]
            ans.Jac = jac
            ans.label = D.label[i]
    if (query.label != ans.label):
        errorRate = errorRate + 1

print("errorRate: ", errorRate / len(Q.index))
