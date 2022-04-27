import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from typing import *
from collections import defaultdict
from handle import handle
from clean import clean


def recall_calculation(predict: list, gnd: pd.DataFrame):
    cnt = 0
    for i in range(len(predict)):
        if not gnd[(gnd['lid'] == predict[i][0]) & (gnd['rid'] == predict[i][1])].empty:
            cnt += 1
    return cnt / gnd.values.shape[0]


def x1_test(data: pd.DataFrame, limit: int, model_path: str) -> list:
    # clusters = handle(data)
    features=clean(data)
    model = SentenceTransformer(model_path, device='cpu')
    encodings = model.encode(sentences=data['title'], batch_size=256, normalize_embeddings=True)
    topk = 50
    candidate_pairs: List[Tuple[int, int, float]] = []
    ram_capacity_list=features['ram_capacity'].values
    cpu_model_list=features['cpu_model'].values
    buckets=defaultdict(list)
    for idx in range(data.shape[0]):
        brands=features['brand'][idx]
        for brand in brands:
            buckets[brand].append(idx)
        if len(brands)==0:
            buckets['0'].append(idx)
    visited_set = set()
    ids = data['id'].values
    # for key in clusters:
    #     cluster = clusters[key]
    for key in buckets:
        cluster=buckets[key]
        embedding_matrix = encodings[cluster]
        k = min(topk, len(cluster))
        index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
        index_model.hnsw.efConstruction = 100
        index_model.add(embedding_matrix)
        index_model.hnsw.efSearch = 256
        D, I = index_model.search(embedding_matrix, k)
        for i in range(len(D)):
            for j in range(len(D[0])):
                index1=cluster[i]
                index2=cluster[I[i][j]]
                s1 = ids[index1]
                s2 = ids[index2]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = str(small) + " " + str(large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                if not (ram_capacity_list[index1]=='0' or ram_capacity_list[index2]=='0' or ram_capacity_list[index1]==ram_capacity_list[index2]):
                    continue
                intersect=cpu_model_list[index1].intersection(cpu_model_list[index2])
                if not (len(cpu_model_list[index1])==0 or len(cpu_model_list[index2])==0 or len(intersect)!=0):
                    continue
                candidate_pairs.append((small, large, D[i][j]))

    # x11_pairs = block_with_attr(raw_data, attr="title")
    # for (x1, x2) in x11_pairs:
    #     vector1 = encodings[x1]
    #     vector2 = encodings[x2]
    #     distance = 0
    #     for k in range(len(vector1)):
    #         distance += (vector1[k]-vector2[k])**2
    #     small=min(ids[x1],ids[x2])
    #     large=max(ids[x1],ids[x2])
    #     visit_token=str(small)+" "+str(large)
    #     if visit_token not in visited_set:
    #         candidate_pairs.append((small, large, distance))
    #         visited_set.add(visit_token)

    candidate_pairs.sort(key=lambda x: x[2])
    candidate_pairs = candidate_pairs[:limit]

    output = list(map(lambda x: (x[0], x[1]), candidate_pairs))

    # predict = output
    # # cnt = 0
    # gnd = pd.read_csv("../Y1.csv")
    # for it in gnd:
    #     if it not in predict:
    #         print(it)
    # for i in range(len(predict)):
    #     if not gnd[(gnd['lid'] == predict[i][0]) & (gnd['rid'] == predict[i][1])].empty:
    #         cnt += 1
    # recall = cnt / gnd.values.shape[0]
    # print('recall:\t', recall)
    return output


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
    # jaccard_similarities = []
    # candidate_pairs_real_ids = []
    # for it in tqdm(candidate_pairs):
    #     id1, id2 = it
    #     # get real ids
    #     real_id1 = X['id'][id1]
    #     real_id2 = X['id'][id2]
    #     if id1 < id2:  # NOTE: This is to make sure in the final output.csv, for a pair id1 and id2 (assume id1<id2), we only include (id1,id2) but not (id2, id1)
    #         candidate_pairs_real_ids.append((id1, id2))
    #     else:
    #         candidate_pairs_real_ids.append((id1, id2))

        # compute jaccard similarity
    #     name1 = str(X[attr][id1])
    #     name2 = str(X[attr][id2])
    #     s1 = set(name1.lower().split())
    #     s2 = set(name2.lower().split())
    #     jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    # candidate_pairs_real_ids = [x for _, x in sorted(zip(jaccard_similarities, candidate_pairs_real_ids), reverse=True)]
    return candidate_pairs


def save_output(X1_candidate_pairs,
                X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csvcpu_model=set(list(map(lambda x:x[1:],cpu_model_list)))
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


if __name__ == '__main__':

    path = './fromstart_further_x1_berttiny_finetune_epoch20_margin0.01'
    mode = 0
    if mode == 0:
        raw_data = pd.read_csv("X1.csv")
        raw_data['title'] = raw_data.title.str.lower()
        x1_pairs = x1_test(raw_data, 1000000, path)
        raw_data = pd.read_csv("X2.csv")
        save_output(x1_pairs, [])
        print("success")
        # calculate
        # with open('../Y1.csv', 'r') as csv1, open('output.csv', 'r') as csv2:
        #     import1 = csv1.readlines()
        #     import2 = csv2.readlines()
        #     same = 0
        #     for row in import2:
        #         if row in import1:
        #             same = same + 1
        #     print(same / len(import1)*2815)

    elif mode == 1:
        test_data = pd.read_csv("../x1_test.csv")
        train_data = pd.read_csv("../x1_train.csv")
        origin_data = pd.read_csv("../X1.csv")
        test_gnd = pd.read_csv("../y1_test.csv")
        train_gnd = pd.read_csv("../y1_train.csv")
        origin_gnd = pd.read_csv("../Y1.csv")
        test_data['title'] = test_data.title.str.lower()
        train_data['title'] = train_data.title.str.lower()
        origin_data['title'] = origin_data.title.str.lower()
        # test_data['instance_id']=test_data['id']
        # train_data['instance_id']=train_data['id']
        # origin_data['instance_id']=origin_data['id']
        test_data = test_data[['id', 'title']]
        train_data = train_data[['id', 'title']]
        origin_data = origin_data[['id', 'title']]
        test_pairs = x1_test(test_data, 488, path)
        train_pairs = x1_test(train_data, 2326, path)
        origin_pairs = x1_test(origin_data, 2814, path)
        raw_data = pd.read_csv('../X1.csv')
        gnd = pd.read_csv('../Y1.csv')
        gnd['cnt'] = 0
        features=clean(raw_data)
        for idx in range(len(origin_pairs)):
            left_id = origin_pairs[idx][0]
            right_id = origin_pairs[idx][1]
            index = gnd[(gnd['lid'] == left_id) & (gnd['rid'] == right_id)].index.tolist()
            if len(index) > 0:
                if len(index) > 1:
                    raise Exception
                gnd['cnt'][index[0]] += 1
                if gnd['cnt'][index[0]] > 1:
                    print(index)
            else:
                left_text = raw_data[raw_data['id'] == left_id]['title'].values[0]
                right_text = raw_data[raw_data['id'] == right_id]['title'].values[0]
                if left_text != right_text:
                    print(idx, left_id, right_id)
                    print(left_text, '|', right_text)
                    print(features[features['instance_id']==left_id]['brand'].iloc[0],'|||',features[features['instance_id']==right_id]['brand'].iloc[0])
                    print(features[features['instance_id'] == left_id]['family'].iloc[0], '|||',
                          features[features['instance_id'] == right_id]['family'].iloc[0])
                    print(features[features['instance_id'] == left_id]['cpu_model'].iloc[0], '|||',
                          features[features['instance_id'] == right_id]['cpu_model'].iloc[0])
                    print(features[features['instance_id']==left_id]['pc_name'].iloc[0],'|||',features[features['instance_id']==right_id]['pc_name'].iloc[0])
                pass
        print('-----------------------------------------------------------------------------------------------')
        left = gnd[gnd['cnt'] == 0]
        for idx in left.index:
            # if features[features['instance_id']==left['lid'][idx]]['brand'].iloc[0]!=features[features['instance_id'] == left['rid'][idx]]['brand'].iloc[0]:
                print(left['lid'][idx], ',', left['rid'][idx])
                print(raw_data[raw_data['id'] == left['lid'][idx]]['title'].iloc[0], '|||',
                      raw_data[raw_data['id'] == left['rid'][idx]]['title'].iloc[0])
        print("Model: %s, test recall: %f, train recall: %f, origin recall: %f" % (
            path, recall_calculation(test_pairs, test_gnd), recall_calculation(train_pairs, train_gnd),
            recall_calculation(origin_pairs, origin_gnd)))
