import pandas as pd
from typing import *
from BackUnion import back_union
from FeatureExtracting import extract_x1, extract_x2
from EntityBlocking import block_x1, block_x2

nonsense = ['|', ',', '-', ':', '/', '+', '&']


def cal_recall(data, gnd):
    predict_pd = pd.read_csv('output.csv')
    # unions = back_union(gnd)
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
    # left = gnd[gnd['cnt'] == 0].reset_index()
    # with open('records/weird.txt', 'w', encoding='utf-8') as file:
    #     for idx in range(left.shape[0]):
    #         print(left['lid'][idx], ',', left['rid'][idx])
    #         print(data[data['id'] == left['lid'][idx]].iloc[0])
    #         print(data[data['id'] == left['rid'][idx]].iloc[0])
    #         print()
    #         for union in unions:
    #             if left['lid'][idx] in union or left['rid'][idx] in union:
    #                 file.write(str(union) + '\n')
    #                 for it in union:
    #                     file.write(data[data['id'] == it]['title'].iloc[0] + '\n')
    #                 file.write('\n')
    #                 unions.remove(union)
    #                 break


def bucket_blocking(data) -> List[Tuple[Set[str], List[int]]]:
    buckets: List[Tuple[Set[str], List[int]]] = []
    for i in range(data.shape[0]):
        max_similarity = 0
        target_bucket = None
        vector = set([it for it in data['title'][i].lower().split() if it not in nonsense])
        for bucket in buckets:
            jaccard_similarity = len(vector.intersection(bucket[0])) / max(len(vector), len(bucket[0]))
            if jaccard_similarity > max_similarity and jaccard_similarity >= 0.5:
                max_similarity = jaccard_similarity
                target_bucket = bucket
        if max_similarity > 0:
            target_bucket[1].append(data['id'][i])
        else:
            buckets.append((vector, [data['id'][i]]))
    return buckets


def gen_candidates(buckets) -> []:
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
    return candidates_pair


def save_output(pairs_x1, expected_size_x1, pairs_x2, expected_size_x2):
    if len(pairs_x1) > expected_size_x1:
        pairs_x1 = pairs_x1[:expected_size_x1]
    elif len(pairs_x1) < expected_size_x1:
        pairs_x1.extend([(0, 0)] * (expected_size_x1 - len(pairs_x1)))
    if len(pairs_x2) > expected_size_x2:
        pairs_x2 = pairs_x2[:expected_size_x2]
    elif len(pairs_x2) < expected_size_x2:
        pairs_x2.extend([(0, 0)] * (expected_size_x2 - len(pairs_x2)))
    output = pairs_x1 + pairs_x2
    output_df = pd.DataFrame(output, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)


if __name__ == '__main__':
    raw_data = pd.read_csv('X1.csv')
    gnd = pd.read_csv('Y1.csv')
    dataset = extract_x1(raw_data)
    candidates_x1 = block_x1(dataset)
    # cal_recall(raw_data, gnd)

    candidates_x2 = [(0, 0)] * 2000000
    save_output(candidates_x1, 1000000, candidates_x2, 2000000)
    # raw_data = pd.read_csv('X2.csv')
    # gnd = pd.read_csv('Y2.csv')
    # dataset = extract_x2(raw_data)
    # candidates = block_x2(dataset)
    # candidates.to_csv('output.csv', index=0)
    # cal_recall(raw_data, gnd)

    # id_buckets = bucket_blocking(raw_data)
    # candidates = gen_candidates(id_buckets)
    # expected_cand_size = 1000000
    # output(candidates, expected_cand_size)
