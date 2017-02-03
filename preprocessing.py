import pandas as pd
import numpy as np

class Bound:
    def __init__(self, df):
        self.tmax = df.index.max()
        self.tmin = df.index.min()
        self.xmax = df.max().max()
        self.xmin = df.min().min()

def readDataasDF(file_path):
    df = pd.read_csv(file_path, header=None)
    df.rename(columns={0: 'label'}, inplace=True)
    return df
# data: S&P500_08to10, row(index): date, columns: ticker
# del D['date'] # date를 index로 지정한 후, date열 제거

def TimeSeriesTranstoSet(q, BD, e, s):
    r = np.floor((q - BD.xmin)/s + 1)
    c = globals()['col']
    query = sorted(set((r-1) * COLUMN_NUM + c))
    return query

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
