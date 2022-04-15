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
    brands = ['sandisk', 'lexar', 'kingston', 'intenso', 'toshiba', 'sony', 'pny', 'samsung', '']
    X = pd.read_csv('X2.csv')
    Y = pd.read_csv('Y2.csv')
    buckets = back_union(Y)
    results = []
    for bucket in buckets:
        contents = []
        for ele in bucket:
            contents.append(X[X['id'] == ele]['name'].iloc[0])
        for brand in brands:
            if brand in contents[0].lower():
                results.append((brand, bucket, contents))
                break
    results.sort(key=lambda x: x[0])
    for result in results:
        print()
        print(result[1])
        for content in result[2]:
            print(content)
        print()
