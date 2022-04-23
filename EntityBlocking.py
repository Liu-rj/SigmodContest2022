import queue
from collections import defaultdict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import *

from scipy.spatial import distance

nonsense = ['|', ',', '-', ':', '/', '+', '&']


def block_x2(dataset: pd.DataFrame):
    """
    Give an identification for each record according to their cleaned field values
    and match records based on their identification

    param RF_dataset: feature Dataframe of X2 after extracting

    :return:
            A DataFrame of matched pairs which contains following columns:
            {left_instance_id: the left instance of a matched pair
             left_instance_id: the right instance of a matched pair}
    """
    model_2_type = {"3502450": "rainbow", "3502460": "rainbow", "3502470": "rainbow", "3502480": "rainbow",
                    "3502490": "rainbow", "3502491": "rainbow",
                    "3503460": "basic", "3503470": "basic", "3503480": "basic", "3503490": "basic",
                    "3533470": "speed", "3533480": "speed", "3533490": "speed", "3533491": "speed", "3533492": "speed",
                    "3534460": "premium", "3534470": "premium", "3534480": "premium", "3534490": "premium",
                    "3534491": "premium",
                    "3521451": "alu", "3521452": "alu", "3521462": "alu", "3521471": "alu",
                    "3521472": "alu", "3521481": "alu", "3521482": "alu", "3521480": "alu", "3521491": "alu",
                    "3521492": "alu",
                    "3511460": "business", "3511470": "business", "3511480": "business", "3511490": "business",
                    "3500450": "micro", "3500460": "micro", "3500470": "micro", "3500480": "micro",
                    "3523460": "mobile", "3523470": "mobile", "3523480": "mobile",
                    "3524460": "mini", "3524470": "mini", "3524480": "mini",
                    "3531470": "ultra", "3531480": "ultra", "3531490": "ultra", "3531491": "ultra", "3531492": "ultra",
                    "3531493": "ultra",
                    "3532460": "slim", "3532470": "slim", "3532480": "slim", "3532490": "slim", "3532491": "slim",
                    "3537490": "highspeed", "3537491": "highspeed", "3537492": "highspeed",
                    "3535580": "imobile", "3535590": "imobile",
                    "3536470": "cmobile", "3536480": "cmobile", "3536490": "cmobile",
                    "3538480": "flash", "3538490": "flash", "3538491": "flash"}

    model = SentenceTransformer('model/282', device='cpu')
    encodings = model.encode(sentences=dataset['name'], batch_size=256, normalize_embeddings=True)

    sony_capacity_single = ["1tb", "256gb"]
    sony_capacity_memtype_type = ["32gb", "4gb"]

    buckets: Dict[str, List] = defaultdict(list)
    mid_buckets = []
    large_buckets = []
    unidentified: List[Tuple[Tuple, int]] = []

    ids = dataset['id'].values
    series = dataset['series'].values
    pat_hb = dataset['pat_hb'].values
    for idx in range(dataset.shape[0]):
        # instance_id = RF_dataset['id'][idx]
        brand = dataset['brand'][idx]
        capacity = dataset['capacity'][idx]
        mem_type = dataset['mem_type'][idx]
        product_type = dataset['type'][idx]
        model = dataset['model'][idx]
        # item_code = RF_dataset['item_code'][idx]
        name = ''.join(sorted(dataset['name'][idx].split()))

        buckets[name].append(idx)

        if product_type == '0' and brand == "intenso" and model in model_2_type.keys():
            product_type = model_2_type[model]

        # if capacity in ('256g', '512g', '1t', '2t') and brand not in ('samsung', 'sandisk'):
        #     high_priority[f'{brand}.{capacity}'].append((instance_id, name))
        #     continue

        key = f'{brand}.{mem_type}.{capacity}.{model}.{product_type}'
        # sig = 'high'
        # if capacity == '0' and model == '0' and product_type == '0':
        #     key, sig = gen_key_sig(key, pat_hb, series)
        # elif capacity == '0' and model == '0':
        #     key, sig = gen_key_sig(key, pat_hb, series)
        # elif model == '0' and product_type == '0':
        #     key, sig = gen_key_sig(key, pat_hb, series)
        # elif product_type == '0':
        #     key, sig = gen_key_sig(key, pat_hb, series)
        # if brand == '0' and sig == 'mid':
        #     sig = 'low'
        # if sig == 'high':
        #     high_priority[key].append(instance_id)
        # elif sig == 'mid':
        #     mid_priority[key].append(instance_id)
        # else:
        #     low_priority[key].append(instance_id)
        buckets[key].append(idx)

    candidates = set()
    for key in buckets.keys():
        bucket = buckets[key]
        if 20 < len(bucket) < 100:
            mid_buckets.append(bucket)
        elif len(bucket) > 100:
            large_buckets.append(bucket)
        else:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    s1 = ids[bucket[i]]
                    s2 = ids[bucket[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    candidates.add((small, large))

    visited_set = set()
    mid_pairs = []
    for bucket in mid_buckets:
        embedding_matrix = encodings[bucket]
        distance_matrix = distance.cdist(embedding_matrix, embedding_matrix, metric='cosine')
        idx = np.argsort(distance_matrix)[:, :len(bucket)]
        for i in range(len(idx)):
            for j in range(len(idx[0])):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[idx[i][j]]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set or visit_token in candidates:
                    continue
                visited_set.add(visit_token)
                mid_pairs.append((small, large, distance_matrix[i][idx[i][j]]))
    mid_pairs.sort(key=lambda x: x[2])
    mid_pairs = mid_pairs[:50000]
    mid_pairs = [(x[0], x[1]) for x in mid_pairs]

    large_pairs = []
    for bucket in large_buckets:
        embedding_matrix = encodings[bucket]
        index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
        index_model.hnsw.efConstruction = 100
        index_model.add(embedding_matrix)
        index_model.hnsw.efSearch = 256
        D, I = index_model.search(embedding_matrix, 50)
        for i in range(len(D)):
            for j in range(len(D[0])):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[I[i][j]]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set or visit_token in candidates:
                    continue
                visited_set.add(visit_token)
                large_pairs.append((small, large, D[i][j]))
        # new_buckets: Dict[str, List] = defaultdict(list)
        # for item in bucket:
        #     new_buckets[f'{series[item]}.{pat_hb[item]}'].append(item)
        # for new_buck in new_buckets:
        #     if len(new_buck) < 20:
        #         for i in range(len(new_buck)):
        #             for j in range(i + 1, len(new_buck)):
        #                 s1 = ids[new_buck[i]]
        #                 s2 = ids[new_buck[j]]
        #                 small = min(s1, s2)
        #                 large = max(s1, s2)
        #                 candidates.add((small, large))
        #     else:
        #         embedding_matrix = encodings[bucket]
        #         index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
        #         index_model.hnsw.efConstruction = 100
        #         index_model.add(embedding_matrix)
        #         index_model.hnsw.efSearch = 256
        #         D, I = index_model.search(embedding_matrix, 50)
        #         for i in range(len(D)):
        #             for j in range(len(D[0])):
        #                 s1 = ids[bucket[i]]
        #                 s2 = ids[bucket[I[i][j]]]
        #                 if s1 == s2:
        #                     continue
        #                 small = min(s1, s2)
        #                 large = max(s1, s2)
        #                 visit_token = (small, large)
        #                 if visit_token in visited_set or visit_token in candidates:
        #                     continue
        #                 visited_set.add(visit_token)
        #                 large_pairs.append((small, large, D[i][j]))
    large_pairs.sort(key=lambda x: x[2])
    large_pairs = large_pairs[:2000000 - len(mid_pairs) - len(candidates)]
    large_pairs = [(x[0], x[1]) for x in large_pairs]

    # jaccard_similarities = []
    # s1 = set(bucket[i][1].split())
    # s2 = set(bucket[j][1].lower().split())
    # jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    # candidates = [x for _, x in sorted(zip(jaccard_similarities, candidates), reverse=True)]
    return list(candidates) + mid_pairs + large_pairs
