import pandas as pd
from FeatureExtracting import extract_x2
from EntityBlocking import block_x2
from collections import defaultdict
import re
from tqdm import tqdm
import time
from typing import *


def cal_recall(data, gnd):
    predict_pd = pd.read_csv('output.csv')
    gnd['cnt'] = 0
    predict = predict_pd.values.tolist()
    for idx in range(len(predict)):
        if predict[idx][0] == 0:
            continue
        index = gnd[(gnd['lid'] == predict[idx][0]) & (gnd['rid'] == predict[idx][1])].index.tolist()
        if len(index) > 0:
            gnd['cnt'][index[0]] += 1
        if len(index) > 1:
            raise Exception
    print(sum(gnd['cnt']))
    print(sum(gnd['cnt']) / gnd.values.shape[0])
    # print that are uncovered
    # left = gnd[gnd['cnt'] == 0].reset_index()
    # for idx in range(left.shape[0]):
    #     print(left['lid'][idx], ',', left['rid'][idx])
    #     print(data[data['id'] == left['lid'][idx]]['title'].iloc[0])
    #     print(data[data['id'] == left['rid'][idx]]['title'].iloc[0])
    #     print()


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
    output_df = pd.DataFrame(output, columns=['left_instance_id', 'right_instance_id'])
    output_df.to_csv('output.csv', index=False)


brands_x1 = ['dell', 'lenovo', 'acer', 'asus', 'hp', 'panasonic', 'toshiba', 'sony', 'epson']


def block_with_attr(data, attr):  # replace with your logic.
    """
    This function performs blocking using attr
    :param data: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    """

    # build index from patterns to tuples
    pattern2id_1 = defaultdict(list)
    pattern2id_2 = defaultdict(list)
    pattern2id_3 = defaultdict(list)
    pattern2id_4 = defaultdict(list)
    for i in tqdm(range(data.shape[0])):
        attr_i = str(data[attr][i])
        pattern_1 = " ".join(sorted(attr_i.lower().split()))  # use the whole attribute as the pattern
        pattern2id_1[pattern_1].append(i)

        pattern_2 = re.findall("\w+\s\w+\d+", attr_i)  # look for patterns like "thinkpad x1"
        if len(pattern_2) != 0:
            pattern_2 = list(sorted(pattern_2))
            pattern_2 = [str(it).lower() for it in pattern_2]
            pattern2id_2[" ".join(pattern_2)].append(i)

        pattern_3 = []
        pattern_3_tmp = re.findall("\w+-\w+", attr_i)
        for x in pattern_3_tmp:
            if "bit" in x.lower():
                continue
            pattern_3.append(x)
        if len(pattern_3) != 0:
            pattern_3 = list(sorted(pattern_3))
            pattern_3 = [str(it).lower() for it in pattern_3]
            pattern2id_3[" ".join(pattern_3)].append(i)

        pattern_4_tmp = re.findall("\d+\w+", attr_i)
        pattern_4 = []
        for x in pattern_4_tmp:
            if len(x) >= 4:
                pattern_4.append(x)
        if len(pattern_4) != 0:
            pattern_4 = list(sorted(pattern_4))
            pattern_4 = [str(it).lower() for it in pattern_4]
            pattern2id_4[" ".join(pattern_4)].append(i)
    # add id pairs that share the same pattern to candidate set
    candidate_pairs_1 = []
    for pattern in tqdm(pattern2id_1):
        ids = list(sorted(pattern2id_1[pattern]))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                candidate_pairs_1.append((ids[i], ids[j]))  #
    # add id pairs that share the same pattern to candidate set
    candidate_pairs_2 = []
    for pattern in tqdm(pattern2id_2):
        ids = list(sorted(pattern2id_2[pattern]))
        if len(ids) < 100:  # skip patterns that are too common
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate_pairs_2.append((ids[i], ids[j]))

    candidate_pairs_3 = []
    for pattern in tqdm(pattern2id_3):
        ids = list(sorted(pattern2id_3[pattern]))
        if len(ids) < 100:  # skip patterns that are too common
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate_pairs_3.append((ids[i], ids[j]))

    candidate_pairs_4 = []
    for pattern in tqdm(pattern2id_4):
        ids = list(sorted(pattern2id_4[pattern]))
        if len(ids) < 100:  # skip patterns that are too common
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    candidate_pairs_4.append((ids[i], ids[j]))

    # remove duplicate pairs and take union
    candidate_pairs = set(candidate_pairs_2)
    candidate_pairs = candidate_pairs.union(set(candidate_pairs_1)).union(set(candidate_pairs_3)).union(
        set(candidate_pairs_4))
    candidate_pairs = list(candidate_pairs)

    # sort candidate pairs by jaccard similarity.
    # In case we have more than 1000000 pairs (or 2000000 pairs for the second dataset),
    # sort the candidate pairs to put more similar pairs first,
    # so that when we keep only the first 1000000 pairs we are keeping the most likely pairs
    jaccard_similarities = []
    candidate_pairs_real_ids = []
    for it in tqdm(candidate_pairs):
        id1, id2 = it

        # get real ids
        real_id1 = data['id'][id1]
        real_id2 = data['id'][id2]
        # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2),
        # we only include (id1,id2) but not (id2, id1)
        if real_id1 < real_id2:
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

        # compute jaccard similarity
        name1 = str(data[attr][id1])
        name2 = str(data[attr][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids


if __name__ == '__main__':
    features = pd.read_csv('X1.csv')
    features = pd.read_csv('X2.csv')
    candidates_x2 = block_with_attr(features, attr='name')
    # features['name'] = features.name.str.lower()
    # features = extract_x2(features)
    # candidates_x2 = block_x2(features)
    candidates_x1 = []
    # save_output(candidates_x1, 1000000, candidates_x2, 2000000)
    output_df = pd.DataFrame(candidates_x2, columns=['left_instance_id', 'right_instance_id'])
    output_df.to_csv('output.csv', index=False)

    # gnd_truth = pd.read_csv('Y2.csv')
    # cal_recall(features, gnd_truth)
