import pandas as pd
from typing import *
from mapreduce import *
import time


def save_output(X1_candidate_pairs, X2_candidate_pairs):
    # save the candset for both datasets to a SINGLE file output.csv
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for RF_dataset X1 and 2000000 pairs for RF_dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for RF_dataset X1 and 2000000 pairs for RF_dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend(
            [(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    # make sure to have the pairs in the first RF_dataset first
    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs
    output_df = pd.DataFrame(all_cand_pairs, columns=[
        "left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for RF_dataset X1, and the remaining pairs are for RF_dataset X2
    output_df.to_csv("output.csv", index=False)


def back_union(Y) -> List[Set]:
    unions: List[Set[int]] = []
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


def cal_recall(gnd):
    predict_pd = pd.read_csv('output.csv')
    gnd['cnt'] = 0
    predict = predict_pd.values.tolist()
    for idx in range(len(predict)):
        if predict[idx][0] == 0:
            break
        index = gnd[(gnd['lid'] == predict[idx][0]) & (
                gnd['rid'] == predict[idx][1])].index.tolist()
        if len(index) > 0:
            gnd['cnt'][index[0]] += 1
        if len(index) > 1:
            print('error')
            exit()
    print(sum(gnd['cnt']))
    print(sum(gnd['cnt']) / gnd.values.shape[0])


def get_x1_pairs(data, embeddings):
    topk = 50
    limit = 1000000
    index = faiss.IndexHNSWFlat(len(embeddings[0]), 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    index.hnsw.efSearch = 512
    D, I = index.search(embeddings, topk)
    shape0 = len(D)
    shape1 = len(D[0])
    hashset = {}
    maxscore = 0
    pq = PriorityQueue()
    for j in range(shape1):
        for i in range(shape0):
            s1 = min(I[i][j], i)
            s2 = max(I[i][j], i)
            token = str(s1) + " " + str(s2)
            if s1 == s2 or token in hashset:
                continue
            if len(hashset) < limit:
                hashset[token] = 1
                pq.put((-D[i][j], s1, s2))
                maxscore = max(maxscore, D[i][j])
            elif D[i][j] < maxscore:
                hashset[token] = 1
                pq.put((-D[i][j], s1, s2))
                entry = pq.get()
                maxscore = -entry[0]
    candidate_set = []
    while not pq.empty():
        score, i, j = pq.get()
        candidate_set.append([i, j])
    output = []
    for tuple in candidate_set:
        lid = data['id'][tuple[0]]
        rid = data['id'][tuple[1]]
        if lid == rid:
            continue
        output.append((min(lid, rid), max(lid, rid)))
        if len(output) > limit:
            break
    return output


# all-MiniLM-L6-v2-> 1: 3854s, 16: 3449s
# paraphrase-MiniLM-L3-v2-> 1: 2318, 2: 2138s
if __name__ == '__main__':
    # start = time.time()

    model_pth = 'model/paraphrase-MiniLM-L3-v2-2022-04-09_13-34-20'
    data_path = 'X1.csv'
    raw_data = pd.read_csv(data_path)
    x1_pairs = run_mapreduce(model_pth=model_pth, data=raw_data, nprocs=16)
    dd = pd.read_csv("X2.csv")
    x2_pairs = []
    save_output(x1_pairs, x2_pairs)

    # end = time.time()
    # print('time:', end - start)
    # ground = pd.read_csv('Y1.csv')
    # cal_recall(ground)
