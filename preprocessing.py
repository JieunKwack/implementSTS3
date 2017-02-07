import pandas as pd
import numpy as np

class Bound:
    def __init__(self, df):
        self.tmax = df.columns.values[1:].max()
        self.tmin = df.columns.values[1:].min()
        self.xmax = df.max().max()
        self.xmin = df.min().min()
        self.rows = 0
        self.columns = 0

    def setRowsAndCols(self, rows, cols):
        self.rows = rows
        self.cols = cols

class Bound_q:
    def __init__(self,df):
        self.tmax = df.index.values[df.index!='label'].max()
        self.tmin = df.index.values[df.index!='label'].min()
        self.xmax = df.max().max()
        self.xmin = df.min().min()

class NN:
    def __init__(self):
        self.TS = 0
        self.Jac = 0
        self.label = ''

def readDataasDF(file_path):
    df = pd.read_csv(file_path, header=None)
    df.rename(columns={0: 'label'}, inplace=True)
    return df

def TimeSeriesTranstoSet(S, bound, e, s):
    COLUMN_NUM = ((bound.tmax-bound.tmin)/e)
    row = (S.loc[:,S.columns!='label'] - bound.xmin)/s + 1
    col = ((S.columns.values[1:] - bound.tmin)/e + 1)
    number = ((row-1).mul(COLUMN_NUM) + col).astype(int)
    bound.setRowsAndCols(row.max().max(), col.max().max())
    result = pd.DataFrame(data=(sorted(set(number.loc[element])) for element in number.index))
    result.to_csv('TimeSeriesTranstoSet.csv')
    return result

def QueryTranstoSet(Q, bound, e, s):
    COLUMN_NUM = ((bound.tmax-bound.tmin)/e)
    row = (Q[Q.index!='label'] - bound.xmin)/s + 1
    col = ((Q.index.values[Q.index!='label'] - bound.tmin)/e + 1)
    number = ((row-1).mul(COLUMN_NUM) + col).astype(int)
    result = pd.DataFrame(sorted(set(number)))
    result.to_csv('QueryTranstoSet.csv', mode="a")
    return sorted(set(number))

def Trans_outQuery_to_Set(Q, bound, e, s):
    Q_in, Q_out = divideQ_to_Qin_and_out(Q, bound)# divide Q to Q_in and Q_out
    Qin = QueryTranstoSet(Q_in, bound, e, s)
    BQ = Bound_q(Q_out)
    Qout = QueryTranstoSet(Q_out, BQ, e, s)
    maxNumber = bound.rows * bound.columns # maximal cell ID in Bound(D)
    Qout = Qout + maxNumber
    Q_trans = set(Qin).union(set(Qout))
    return sorted(Q_trans)

def divideQ_to_Qin_and_out(q, bound):
    q = q[q.index!='label']
    Qin = q.loc[lambda q: q.index[(q > bound.xmin) & (q < bound.xmax)]]
    Qout = q.loc[lambda q: q.index[(q > bound.xmax) | (q < bound.xmin)]]
    return Qin, Qout

def Jaccard(d, q):
    U = q.union(d)
    I = q.intersection(d)
    return len(I)/len(U)
