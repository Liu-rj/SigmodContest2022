import csv

from handler import handle
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import re

Flag = True

#
#
def block_with_attr(X, attr):  # replace with your logic.
    '''
    This function performs blocking using attr
    :param X: dataframe
    :param attr: attribute used for blocking
    :return: candidate set of tuple pairs
    '''

    # build index from patterns to tuples
    pattern2id_1 = defaultdict(list)
    pattern2id_2 = defaultdict(list)
    pattern2id_3 = defaultdict(list)
    pattern2id_4 = defaultdict(list)
    for i in tqdm(range(X.shape[0])):
        attr_i = str(X[attr][i])
        pattern_1 = " ".join(sorted(attr_i.lower().split()))  # use the whole attribute as the pattern
        pattern2id_1[pattern_1].append(i)

        pattern_2 = re.findall("\w+\s\w+\d+", attr_i)  # look for patterns like "thinkpad x1"
        if len(pattern_2) != 0:
            pattern_2 = list(sorted(pattern_2))
            pattern_2 = [str(it).lower() for it in pattern_2]
            pattern2id_2[" ".join(pattern_2)].append(i)

        pattern_3=[]
        pattern_3_tmp = re.findall("\w+-\w+",attr_i)
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
            pattern_4= list(sorted(pattern_4))
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
    X11_candidate_pairs = []
    if attr == "title":
        data = pd.read_csv(file_name)
        X11_candidate_pairs = handle(data).values.tolist()
        X11_candidate_pairs = map(lambda x: tuple(x), X11_candidate_pairs)
    candidate_pairs = set(candidate_pairs_2)
    candidate_pairs = candidate_pairs.union(set(candidate_pairs_1)).union(set(candidate_pairs_3)).union(
        set(candidate_pairs_4)).union(set(X11_candidate_pairs))
    candidate_pairs = list(candidate_pairs)

    # sort candidate pairs by jaccard similarity.
    # In case we have more than 1000000 pairs (or 2000000 pairs for the second dataset),
    # sort the candidate pairs to put more similar pairs first,
    # so that when we keep only the first 1000000 pairs we are keeping the most likely pairs
    jaccard_similarities = []
    candidate_pairs_real_ids = []
    solved = []
    for i in range(X.shape[0]):
        solved.append(0)
    for it in tqdm(candidate_pairs):
        id1, id2 = it
        solved[id1] = 1
        solved[id2] = 1
        # get real ids
        real_id1 = X['id'][id1]
        real_id2 = X['id'][id2]
        if real_id1 < real_id2:  # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2), we only include (id1,id2) but not (id2, id1)
            candidate_pairs_real_ids.append((real_id1, real_id2))
        else:
            candidate_pairs_real_ids.append((real_id2, real_id1))

        # compute jaccard similarity
        name1 = str(X[attr][id1])
        name2 = str(X[attr][id2])
        s1 = set(name1.lower().split())
        s2 = set(name2.lower().split())
        jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    for i in range(len(solved)):
        if solved[i] == 0:
            print(X['title'][i])
    candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs_real_ids


def save_output(X1_candidate_pairs,
                X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000
    print(len(X1_candidate_pairs))
    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    # if len(X1_candidate_pairs) < expected_cand_size_X1+expected_cand_size_X2:
    #     X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1+expected_cand_size_X2-len(X1_candidate_pairs)))
    # if len(X2_candidate_pairs) < expected_cand_size_X2:
    #     X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


if __name__ == '__main__':
    X1_candidate_pairs = []
    X2_candidate_pairs = []
    for file_name in ['X1.csv']:
        if file_name == "X1.csv":
            X1 = pd.read_csv("X1.csv")
            X1_candidate_pairs = handle(X1).values.tolist()
            #X1_candidate_pairs = block_with_attr(X1, attr="title")


        else:
            # read the datasets
            X2 = pd.read_csv("X2.csv", dtype=object)
            # perform blocking
            X2_candidate_pairs = block_with_attr(X2, attr="name")

            # save results
        save_output(X1_candidate_pairs, [])
        # if Flag:
        #         output.to_csv("Y1.csv", sep=',', encoding='utf-8', index=False)
        #         Flag = False

    #calculate
    with open('../Y1.csv', 'r') as csv1, open('output.csv', 'r') as csv2:
        import1 = csv1.readlines()
        import2 = csv2.readlines()
        same = 0
        for row in import2:
            if row in import1:
                same = same + 1
        print(same / len(import1))
