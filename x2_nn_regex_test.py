import os

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from scipy.spatial import distance
from FeatureExtracting import extract_x2
from typing import *
from collections import defaultdict


def x2_test(data: pd.DataFrame, gnd: pd.DataFrame, limit: int, model_path: str) -> float:
    features = extract_x2(data)

    model = SentenceTransformer(model_path, device='cpu')
    encodings = model.encode(sentences=features['name'], batch_size=256, normalize_embeddings=True)

    topk = 50

    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    unidentified: List[Tuple[Tuple, int]] = []
    for idx in range(features.shape[0]):
        instance_id = features['id'][idx]
        brand = features['brand'][idx]
        capacity = features['capacity'][idx]
        mem_type = features['mem_type'][idx]
        name = features['name'][idx]

        buckets[tuple(name)].append(idx)

        if brand != '0' and mem_type != '0' and capacity != '0':
            buckets[(brand, mem_type, capacity)].append(idx)
        else:
            key = []
            if brand != '0':
                key.append(brand)
            if mem_type != '0':
                key.append(mem_type)
            if capacity != '0':
                key.append(capacity)
            key = tuple(key)
            found = False
            for ele in buckets:
                if set(key).issubset(set(ele)):
                    buckets[ele].append(idx)
                    found = True
            if not found:
                unidentified.append((key, idx))

    for it in unidentified:
        key = it[0]
        found = False
        for ele in buckets:
            if set(key).issubset(set(ele)):
                buckets[ele].append(it[1])
                found = True
        if not found:
            buckets[key].append(it[1])

    candidate_pairs: List[Tuple[int, int, float]] = []

    visited_set = set()
    for key in buckets:
        cluster = buckets[key]
        embedding_matrix = encodings[cluster]
        k = min(topk, len(cluster))
        if len(cluster) < 10000:
            # cosine distance, smaller is better, 1-x*y; ||x-y||=||x||+||y||-2x*y
            distance_matrix = distance.cdist(embedding_matrix, embedding_matrix, metric='cosine')
            idx = np.argsort(distance_matrix)[:, :k]
            for i in range(len(idx)):
                for j in range(len(idx[0])):
                    s1 = data['instance_id'].loc[cluster[i]]
                    s2 = data['instance_id'].loc[cluster[idx[i][j]]]
                    if s1 == s2:
                        continue
                    small = min(s1, s2)
                    large = max(s1, s2)
                    visit_token = str(small) + " " + str(large)
                    if visit_token in visited_set:
                        continue
                    visited_set.add(visit_token)
                    candidate_pairs.append((small, large, 2 * distance_matrix[i][idx[i][j]]))
        else:
            index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
            index_model.hnsw.efConstruction = 100
            index_model.add(embedding_matrix)
            index_model.hnsw.efSearch = 256
            D, I = index_model.search(embedding_matrix, k)
            for i in range(len(D)):
                for j in range(len(D[0])):
                    s1 = data['instance_id'].loc[cluster[i]]
                    s2 = data['instance_id'].loc[cluster[I[i][j]]]
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
    predict = output
    cnt = 0
    for i in range(len(predict)):
        if not gnd[(gnd['lid'] == predict[i][0]) & (gnd['rid'] == predict[i][1])].empty:
            cnt += 1
    recall = cnt / gnd.values.shape[0]
    return recall


def run_test(model_path):
    print(f'Model:\t\t\t {model_path}')
    test_limit = 1357
    train_limit = 3035
    origin_limit = 4392

    test_recall = x2_test(test_data, test_gnd, test_limit, model_path)
    train_recall = x2_test(train_data, train_gnd, train_limit, model_path)
    origin_recall = x2_test(origin_data, origin_gnd, origin_limit, model_path)
    print(f'Test Recall:\t\t {test_recall}')
    print(f'Train Recall:\t\t {train_recall}')
    print(f'Origin X2 Recall:\t {origin_recall}')


if __name__ == '__main__':
    test_data = pd.read_csv("data/x2_test.csv")
    train_data = pd.read_csv("data/x2_train.csv")
    origin_data = pd.read_csv("X2.csv")
    test_gnd = pd.read_csv("data/y2_test.csv")
    train_gnd = pd.read_csv("data/y2_train.csv")
    origin_gnd = pd.read_csv("Y2.csv")
    origin_data['name'] = origin_data.name.str.lower()
    test_data['name'] = test_data.title.str.lower()
    train_data['name'] = train_data.title.str.lower()
    test_data['instance_id'] = test_data['id']
    train_data['instance_id'] = train_data['id']
    origin_data['instance_id'] = origin_data['id']

    nn_model_path = f'model/checkpoints/sts-mix_base/'
    direct = os.listdir(nn_model_path)
    for d in direct:
        path = nn_model_path + d
        run_test(path)
