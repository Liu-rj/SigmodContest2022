import pandas as pd
from typing import *


def back_union(Y) -> List[Set]:
    unions: List[Set] = []
    for i in range(Y.shape[0]):
        found = False
        candidates = []
        for union in unions:
            if Y['lid'][i] in union or Y['rid'][i] in union:
                union.add(Y['lid'][i])
                union.add(Y['rid'][i])
                found = True
                candidates.append(union)
        if len(candidates) > 1:
            union = set()
            for candidate in candidates:
                unions.remove(candidate)
                union = union.union(candidate)
            unions.append(union)
        if not found:
            unions.append({Y['lid'][i], Y['rid'][i]})
    return unions


if __name__ == '__main__':
    X = pd.read_csv('X1.csv')
    Y = pd.read_csv('Y1.csv')
    unions = back_union(Y)
    for union in unions:
        print(union)
        for ele in union:
            print(X[X['id'] == ele]['title'].iloc[0])
        print()
