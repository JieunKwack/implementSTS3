import pandas as pd
import numpy as np

class Bound:
    def __init__(self,D):
        self.tmax = pd.to_datetime(str(D.index.max()), format='%Y%m%d') + np.timedelta64(3, 'D') # 나눠지지 않아서 임의적으로 3만큼 더 키움
        self.tmin = pd.to_datetime(str(D.index.min()), format='%Y%m%d')
        self.xmax = D.max().max()
        self.xmin = D.min().min()

def TimeSeriesTranstoSet(q, BD, e, s):
    r = np.floor((q - BD.xmin)/s + 1)
    c = globals()['col']
    query = sorted(set((r-1) * COLUMN_NUM + c))
    return query
