import pandas as pd
from typing import *
from collections import defaultdict


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


def print_portion(raw_data, gnd, brand_list):
    cnt_dict: Dict[str, int] = defaultdict(int)
    for i in range(gnd.shape[0]):
        left_sentence = raw_data[raw_data['id'] == gnd['lid'][i]]['name'].iloc[0]
        right_sentence = raw_data[raw_data['id'] == gnd['rid'][i]]['name'].iloc[0]
        # print(left_sentence, '|', right_sentence)
        for b in brand_list:
            if b in left_sentence or b in right_sentence:
                cnt_dict[b] += 1
                if b == '':
                    print(left_sentence, '|', right_sentence)
                break
    for key in cnt_dict.keys():
        cnt_dict[key] = cnt_dict[key] / gnd.shape[0]
    print(cnt_dict)


if __name__ == '__main__':
    brands = ['sandisk', 'lexar', 'kingston', 'intenso', 'toshiba', 'sony', 'pny', 'samsung', 'transcend', '']
    X = pd.read_csv('X2.csv')
    Y = pd.read_csv('Y2.csv')
    # X['name'] = X.name.str.lower()

    # print_portion(X, Y, brands)

    buckets = back_union(Y)
    results = []
    for bucket in buckets:
        contents = []
        for ele in bucket:
            contents.append((X[X['id'] == ele]['name'].iloc[0], X[X['id'] == ele]['price'].iloc[0]))
        for brand in brands:
            if brand in contents[0][0].lower():
                results.append((brand, bucket, contents))
                break
    results.sort(key=lambda x: x[0])
    for result in results:
        if result[0] != 'samsung':
            continue
        print()
        print(result[1])
        for content in result[2]:
            print(content[0])
        print()
