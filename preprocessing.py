import pandas as pd
import numpy as np

class Bound:
    def __init__(self, df):
        self.tmax = df.index.max()
        self.tmin = df.index.min()
        self.xmax = df.max().max()
        self.xmin = df.min().min()

class Bound_q:
    def __init__(self,df):
        self.tmax = df.index.values[1:].max()
        self.tmin = df.index.values[1:].min()
        self.xmax = df.max().max()
        self.xmin = df.min().min()

def readDataasDF(file_path):
    df = pd.read_csv(file_path, header=None)
    df.rename(columns={0: 'label'}, inplace=True)
    return df

def TimeSeriesTranstoSet(S, bound, e, s):
    COLUMN_NUM = ((bound.tmax-bound.tmin)/e)
    row = (S.loc[:,S.columns!='label'] - bound.xmin)/s + 1
    col = ((S.columns.values[1:]-BD.tmin)/epsilon+1)
    number = (row-1).mul(COLUMN_NUM) + col
    result = np.floor(pd.DataFrame(data=(sorted(set(number.loc[element])) for element in number.index)))
    result.to_csv('TimeSeriesTranstoSet.csv')
    return result

def Trans_outQuery_to_Set(): # Alg. needs to modify to Code
    # !Add: devide Q into Q_in and Q_out
    Qin = TimeSeriesTranstoSet(Q_in, BD, epsilon, sigma)
    BQ = Bound(Q_out)
    Qout = TimeSeriesTranstoSet(Q_out, BQ, epsilon, sigma)
    maxNumber = BD.rows * BD.columns # maximal cell ID in Bound(D)
    # for t in Qout:
    #     t = t + maxNumber
    Qout = Qout + maxNumber
    Q = set(Qin).union(set(Qout))
    return Q

def Jaccard(d, q):
    U = q.union(d)
    I = q.intersection(d)
    return len(I)/len(U)
