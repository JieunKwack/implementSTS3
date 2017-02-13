import pandas as pd
import datetime as dt
import numpy as np
from preprocessing import Bound, readDataasDF, TimeSeriesTranstoSet, QueryTranstoSet, Bound_q, NN, Trans_outQuery_to_Set, Jaccard
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# set parameter cell size
sigma = 0.82
epsilon = 76

# read some database
D = readDataasDF('CinC_ECG_torso\\CinC_ECG_torso_TRAIN')
Q = readDataasDF('CinC_ECG_torso\\CinC_ECG_torso_TEST')

BD = Bound(D) # compute Bound
# avgTS
# label = D.label.unique()
# D = pd.DataFrame(D[D.label == i].sum() for i in label)
# D.label = label

D_trans = TimeSeriesTranstoSet(D, BD, epsilon, sigma)

errorRate = 0
count = 0
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
        Target = set(filter(lambda x: x == x , D_trans.loc[i]))
        jac = Jaccard(set(Target), set(Q_trans))
        if (ans.Jac < jac):
            ans.TS = i # index: D_trans.loc[i]
            ans.Jac = jac
            ans.label = D.label[i]
    if (query.label != ans.label):
        errorRate += 1
        print(q, ans.TS)
        print(int(query.label), ans.label)
        print("jac: ", ans.Jac)

print("errorRate: ", errorRate / len(Q.index))
# print(errorRate)
print("count: ", count)
