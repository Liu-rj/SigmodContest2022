import pandas as pd
from typing import *


def cal_recall():
    predict_pd = pd.read_csv('output.csv')
    gnd = pd.read_csv('Y1.csv')
    gnd['cnt'] = 0
    predict = predict_pd.values.tolist()
    for idx in range(len(predict)):
        index = gnd[(gnd['lid'] == predict[idx][0]) & (gnd['rid'] == predict[idx][1])].index.tolist()
        if len(index) > 0:
            gnd['cnt'][index[0]] += 1
        if len(index) > 1:
            print('error')
            exit()
    print(sum(gnd['cnt']))
    print(sum(gnd['cnt']) / gnd.values.shape[0])
    left = gnd[gnd['cnt'] == 0].reset_index()
    for idx in range(left.shape[0]):
        print(left['lid'][idx], ',', left['rid'][idx])
        print(X[X['id'] == left['lid'][idx]]['title'].iloc[0])
        print(X[X['id'] == left['rid'][idx]]['title'].iloc[0])
        print()


def output(pairs, size):
    if len(pairs) > size:
        pairs = pairs[:size]
    output_df = pd.DataFrame(pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)
    cal_recall()


if __name__ == '__main__':
    nonsense = ['|', ',', '-', ':', '/', '+', '&']
    X = pd.read_csv('X1.csv')
    Y = pd.read_csv('Y1.csv')
    buckets: List[Tuple[Set[str], List[int]]] = []
    for i in range(X.shape[0]):
        max_similarity = 0
        target_bucket = None
        vector = set([it for it in X['title'][i].lower().split() if it not in nonsense])
        for bucket in buckets:
            jaccard_similarity = len(vector.intersection(bucket[0])) / max(len(vector), len(bucket[0]))
            if jaccard_similarity > max_similarity and jaccard_similarity >= 0.5:
                max_similarity = jaccard_similarity
                target_bucket = bucket
        if max_similarity > 0:
            target_bucket[1].append(X['id'][i])
        else:
            buckets.append((vector, [X['id'][i]]))
    for bucket in buckets:
        print(bucket[1])
    candidates_pair = []
    for bucket in buckets:
        if len(bucket[1]) <= 1:
            continue
        for idx_1 in range(len(bucket[1])):
            for idx_2 in range(idx_1 + 1, len(bucket[1])):
                if bucket[1][idx_1] < bucket[1][idx_2]:
                    candidates_pair.append((bucket[1][idx_1], bucket[1][idx_2]))
                else:
                    candidates_pair.append((bucket[1][idx_2], bucket[1][idx_1]))
    expected_cand_size = 1000000
    output(candidates_pair, expected_cand_size)
