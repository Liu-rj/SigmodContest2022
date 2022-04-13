import pandas as pd
from typing import *


def back_union(gnd) -> List[Set]:
    unions: List[Set] = []
    for i in range(gnd.shape[0]):
        found = False
        candidates = []
        for union in unions:
            if gnd['lid'][i] in union or gnd['rid'][i] in union:
                union.add(gnd['lid'][i])
                union.add(gnd['rid'][i])
                found = True
                candidates.append(union)
        if len(candidates) > 1:
            union = set()
            for candidate in candidates:
                unions.remove(candidate)
                union = union.union(candidate)
            unions.append(union)
        if not found:
            unions.append({gnd['lid'][i], gnd['rid'][i]})
    return unions


if __name__ == '__main__':
    X = pd.read_csv('X2.csv')
    Y = pd.read_csv('Y2.csv')
    buckets = back_union(Y)
    for bucket in buckets:
        print(bucket)
        for ele in bucket:
            print(X[X['id'] == ele]['name'].iloc[0])
        print()
