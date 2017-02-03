import pandas as pd
import datetime as dt
import numpy as np
from preprocessing import Bound

# set parameter cell size
epsilon = 21
sigma = 0.18

# read some database
D = readDataasDF('CBF\\CBF_TRAIN')
Q = readDataasDF('CBF\\CBF_TEST')

BD = Bound(D) # compute Bound

#preproc. translate ts to set
COLUMN_NUM = ((BD.tmax-BD.tmin)/epsilon)
row = (D.loc[:,D.columns!='label'] - BD.xmin)/sigma + 1
col = ((D.columns.values[1:]-BD.tmin)/epsilon+1)
number = (row-1).mul(COLUMN_NUM) + col
result = np.floor(pd.DataFrame(data=(sorted(set(number.loc[element])) for element in number.index)))
result.to_csv('TimeSeriesTranstoSet.csv')

Max_jac = 0
# query = TimeSeriesTranstoSet(Q, BD, epsilon, sigma)
query = set(result.loc['A'])
for ticker in result.index:
    #jaccard sim to query~
    target = set(result.loc[ticker])
    U = query.union(target)
    I = query.intersection(target)
    temp = len(I)/len(U)
    if ((temp > Max_jac) & ('A' != ticker)):
        Max_jac = temp
        Max_ticker = ticker

print(Max_jac, Max_ticker)
