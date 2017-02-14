import csv
import pandas as pd
import datetime as dt
import numpy as np
from preprocessing import Bound, readDataasDF, TimeSeriesTranstoSet, QueryTranstoSet, Bound_q, NN, Trans_outQuery_to_Set, Jaccard
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import heapq as hp

# set parameter cell size
sigma = 0.82
epsilon = 76
k = 5

# read some database
D = readDataasDF('CinC_ECG_torso\\CinC_ECG_torso_TRAIN')
Q = readDataasDF('CinC_ECG_torso\\CinC_ECG_torso_TEST')

BD = Bound(D) # compute Bound
# avgTS
# label = D.label.unique()
# D = pd.DataFrame(D[D.label == i].sum() for i in label)
# D.label = label

D_trans = TimeSeriesTranstoSet(D, BD, epsilon, sigma)
errorList = []
errorRate = 0
for q in Q.index:
    query = Q.loc[q]
    heap = []
    # ans = NN()
    BQ = Bound_q(query)
    if ((BQ.xmin < BD.xmin) | (BD.xmax < BQ.xmax)):# check Query bound out of the DB bound
        Q_trans = Trans_outQuery_to_Set(query, BD, epsilon, sigma)
    else:
        Q_trans = QueryTranstoSet(query, BD, epsilon, sigma)
    for i in D_trans.index: #kNNpart
        Q_trans = set(filter(lambda x: x == x , Q_trans)) # remove NAN
        Target = set(filter(lambda x: x == x , D_trans.loc[i]))
        jac = Jaccard(set(Target), set(Q_trans))
        if (len(heap) == k):
            hp.heappushpop(heap, (jac, i, D.label[i]))
        elif (len(heap) < k):
            hp.heappush(heap, (jac, i, D.label[i]))
    kNN_list = sorted(heap, reverse = True)
    if (query.label != kNN_list[0][2]):
        kNN_list.insert(0, query.label) #qeury.label, (jac, index, label)...
        errorList.append(kNN_list)
        errorRate += 1
        # print(query.label, kNN_list[0][2])

with open('errorList.csv','w') as csvfile:
    errWriter = csv.writer(csvfile, delimiter=',')
    # while errorList:
    errWriter.writerows(errorList)

# print("finish sort: ", kNN_list)
print("errorRate: ", errorRate / len(Q.index))
print(errorRate)
# print("count: ", count)
