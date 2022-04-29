from statistics import mode
import torch.multiprocessing as mp
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from queue import *
from typing import *


def map_func(rank, queue: mp.Queue):

    queue.put((rank, embeddings))
    print(f'process {rank} finished')


def reduce_func(nprocs, queue: mp.Queue, data, process_list: List[mp.Process]):
    recv = []
    for i in range(nprocs):
        em_part = queue.get()
        rank = em_part[0]
        process_list[rank].join()
        recv.append(em_part)
    recv.sort(key=lambda x: x[0])
    embeddings: np.ndarray = recv[0][1]
    for em_part in recv[1:]:
        embeddings = np.concatenate((embeddings, em_part[1]), axis=0)
    print(embeddings.shape)
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
            token = str(s1)+" "+str(s2)
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


def encoding(model_pth, data):
    sentences = data['title']
    model = SentenceTransformer(model_pth)
    encodings: np.ndarray = model.encode(
        sentences, show_progress_bar=True, batch_size=128, convert_to_numpy=True, normalize_embeddings=True)
    return encodings


def run_mapreduce(model_pth, data, nprocs) -> List:
    embeddings = encoding(model_pth, data)
    topk = 50
    limit = 1000000
    index = faiss.IndexHNSWFlat(len(embeddings[0]), 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    index.hnsw.efSearch = 512
    D, I = index.search(embeddings, topk)
    queue = mp.Queue()
    step = int(data.shape[0] / nprocs)
    idx = 0
    process_list = []
    for rank in range(nprocs):
        if rank == nprocs - 1:
            process = mp.Process(target=map_func, args=(
                rank, model_pth, data[idx:].reset_index(drop=True), queue))
        else:
            process = mp.Process(target=map_func, args=(
                rank, model_pth, data[idx:idx + step].reset_index(drop=True), queue))
            idx += step
        process_list.append(process)
        process.start()
    return reduce_func(nprocs, queue, data, process_list)
