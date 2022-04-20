import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from FeatureExtracting import extract_x2
from typing import *
from collections import defaultdict


def x2_test(data: pd.DataFrame, limit: int, model_path: str) -> list:
    features = extract_x2(data)

    model = SentenceTransformer(model_path, device='cpu')
    encodings = model.encode(sentences=features['name'], batch_size=256, normalize_embeddings=True)

    topk = 50

    # embedding_matrix = encodings
    # candidate_pairs: List[Tuple[int, int, float]] = []
    # visited_set = set()
    # index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
    # index_model.hnsw.efConstruction = 100
    # index_model.add(embedding_matrix)
    # index_model.hnsw.efSearch = 256
    # D, I = index_model.search(embedding_matrix, topk)
    # for i in range(len(D)):
    #     for j in range(len(D[0])):
    #         s1 = data['instance_id'].loc[i]
    #         s2 = data['instance_id'].loc[I[i][j]]
    #         if s1 == s2:
    #             continue
    #         small = min(s1, s2)
    #         large = max(s1, s2)
    #         visit_token = str(small) + " " + str(large)
    #         if visit_token in visited_set:
    #             continue
    #         visited_set.add(visit_token)
    #         candidate_pairs.append((small, large, D[i][j]))

    same_name: Dict[str, List[int]] = defaultdict(list)
    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    unidentified: List[Tuple[List, int]] = []
    for idx in range(features.shape[0]):
        instance_id = features['id'][idx]
        brand = features['brand'][idx]
        capacity = features['capacity'][idx]
        mem_type = features['mem_type'][idx]
        name = ''.join(sorted(features['name'][idx].split()))

        same_name[name].append(instance_id)

        buckets[capacity].append(idx)

        # if brand != '0' and mem_type != '0' and capacity != '0':
        #     buckets[(brand, mem_type, capacity)].append(idx)
        # else:
        #     key = []
        #     if brand != '0':
        #         key.append(brand)
        #     if mem_type != '0':
        #         key.append(mem_type)
        #     if capacity != '0':
        #         key.append(capacity)
        #     unidentified.append((key, idx))

    # for it in unidentified:
    #     key = it[0]
    #     found = False
    #     for ele in buckets:
    #         if set(key).issubset(set(ele)):
    #             buckets[ele].append(it[1])
    #             found = True
    #     if not found:
    #         buckets[tuple(key)].append(it[1])

    candidate_pairs: List[Tuple[int, int, float]] = []
    visited_set = set()

    for key in same_name:
        bucket = same_name[key]
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                small = min(bucket[i], bucket[j])
                large = max(bucket[i], bucket[j])
                visited_set.add(str(small) + " " + str(large))
                candidate_pairs.append((small, large, 0))

    ids = data['instance_id'].values
    for key in buckets:
        cluster = buckets[key]
        embedding_matrix = encodings[cluster]
        k = min(topk, len(cluster))
        index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
        index_model.hnsw.efConstruction = 100
        index_model.add(embedding_matrix)
        index_model.hnsw.efSearch = 256
        D, I = index_model.search(embedding_matrix, k)
        for i in range(len(D)):
            for j in range(len(D[0])):
                s1 = ids[cluster[i]]
                s2 = ids[cluster[I[i][j]]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = str(small) + " " + str(large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                candidate_pairs.append((small, large, D[i][j]))

    candidate_pairs.sort(key=lambda x: x[2])
    candidate_pairs = candidate_pairs[:limit]
    output = list(map(lambda x: (x[0], x[1]), candidate_pairs))

    # predict = output
    # cnt = 0
    # gnd = pd.read_csv("Y2.csv")
    # for i in range(len(predict)):
    #     if not gnd[(gnd['lid'] == predict[i][0]) & (gnd['rid'] == predict[i][1])].empty:
    #         cnt += 1
    # recall = cnt / gnd.values.shape[0]
    # print('recall:\t', recall)
    return output


def save_output(X1_candidate_pairs, X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
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
    raw_data = pd.read_csv("X1.csv")
    raw_data = pd.read_csv("X2.csv")

    # path = 'model/checkpoints/sts-mix_base/24000'
    # path = 'model/mix_base'
    path = 'model/282'
    raw_data['name'] = raw_data.name.str.lower()
    raw_data['instance_id'] = raw_data['id']
    x2_pair = x2_test(raw_data, 2000000, path)
    save_output([], x2_pair)
