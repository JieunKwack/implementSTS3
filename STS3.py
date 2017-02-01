import pandas as pd
import datetime as dt
import numpy as np
from preprocessing import Bound

#cell size
epsilon = 10
sigma = 6

# read some database
D = pd.read_csv('S&P500_08to10.csv') #dir위치 재조정 필요 or data import 필요
# data row(index): date, columns: ticker
D.set_index(D.date, inplace=True)
del D['date'] # date를 index로 지정한 후, date열 제거
BD = Bound(D) # compute Bound

COLUMN_NUM = ((BD.tmax - BD.tmin)/np.timedelta64(1,'D')).astype(int) / epsilon

row = np.floor((D-BD.xmin)/sigma + 1)

col = np.array([pd.to_datetime(str(x), format='%Y%m%d') for x in D.index.values])
col = ((col - BD.tmin)/dt.timedelta(epsilon) + 1).astype(int)
col_mat = np.repeat(np.matrix(col), row.shape[1], axis=0)

number = (row-1)*COLUMN_NUM + col_mat.T

result = pd.DataFrame(data=(sorted(set(number.ix[:,ticker])) for ticker in number.columns), index=number.columns)
result.to_csv('TimeSeriesTranstoSet.csv')

Max_jac = 0
# query = TimeSeriesTranstoSet(q, BD, epsilon, sigma)
query = set(result.loc['A'])
for ticker in result.index:
    #jaccard sim to query
    target = set(result.loc[ticker])
    U = query.union(target)
    I = query.intersection(target)
    temp = len(I)/len(U)
    if ((temp > Max_jac) & ('A' != ticker)):
        Max_jac = temp
        Max_ticker = ticker

print(Max_jac, Max_ticker)
