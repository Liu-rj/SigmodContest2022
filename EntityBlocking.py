from collections import defaultdict
import faiss
import numpy as np
import torch
import pandas as pd
from typing import *
from scipy.spatial import distance
from CoSent.model import Model, encode, myTokenizer

nonsense = ['|', ',', '-', ':', '/', '+', '&']

brand_portion = {'sandisk': 0.25, 'lexar': 0.192, '0': 0.028, 'kingston': 0.133, 'intenso': 0.07, 'toshiba': 0.155,
                 'sony': 0.123, 'samsung': 0.013, 'pny': 0.034, 'transcend': 0.008}


def block_x2(dataset: pd.DataFrame, max_limit):
    model_path = f'./CoSent/x2_model/base_model_epoch_{38}.bin'
    my_model = Model(config_path='./CoSent/prajjwal1_bert-tiny/config.json',
                     bert_path='./CoSent/prajjwal1_bert-tiny/pytorch_model.bin')
    my_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    tokenizer = myTokenizer('./CoSent/prajjwal1_bert-tiny')
    encodings = encode(model=my_model, sentences=dataset['name'], tokenizer=tokenizer)

    ids = dataset['id'].values
    series_list = dataset['series'].values
    pat_hb_list = dataset['pat_hb'].values
    brand_list = dataset['brand'].values
    capacity_list = dataset['capacity'].values
    mem_list = dataset['mem_type'].values
    product_list = dataset['type'].values
    model_list = dataset['model'].values
    name_list = dataset['name'].values
    item_code_list = dataset['item_code'].values

    features_list = [capacity_list, mem_list, product_list, model_list, series_list, pat_hb_list]
    buckets: Dict[str, List] = defaultdict(list)
    special_buckets: Dict[str, List] = defaultdict(list)

    for idx in range(dataset.shape[0]):
        buckets[brand_list[idx]].append(idx)
        if brand_list[idx] == 'sandisk' and mem_list[idx] in ('microsd', 'sd') and model_list[idx] == 'ultra+' and \
                capacity_list[idx] == '32g':
            special_buckets['accessoires montres ' + mem_list[idx]].append(idx)
        # elif brand_list[idx] == 'sandisk' and mem_list[idx] == 'microsd' and model_list[idx] == 'ext+' and \
        #         capacity_list[idx] == '16g': /////////////// will reduce recall by 0.001
        #     special_buckets['microsdhc sandisk extreme pro ' + capacity_list[idx]].append(idx)
        # elif brand_list[idx] == 'sony' and mem_list[idx] == 'microsd' and capacity_list[idx] == '128g':
        #     special_buckets['sony microsd 128g'].append(idx) /////////////// danger
        # print(brand_list[idx], '|', capacity_list[idx], '|', series_list[idx], '|', mem_list[idx], '|',
        #       product_list[idx], '|', model_list[idx], '|', name_list[idx])

    visited_set = set()
    candidate_pairs = []

    for key in special_buckets.keys():
        bucket = special_buckets[key]
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[j]]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                candidate_pairs.append((small, large, 0))

    faiss_all_pairs = []
    for key in buckets.keys():
        faiss_pairs = []
        bucket = buckets[key]
        embedding_matrix = encodings[bucket]
        index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
        index_model.hnsw.efConstruction = 100
        index_model.add(embedding_matrix)
        index_model.hnsw.efSearch = 256
        D, I = index_model.search(embedding_matrix, 50)
        for i in range(len(D)):
            for j in range(len(D[0])):
                index1 = bucket[i]
                index2 = bucket[I[i][j]]
                s1 = ids[index1]
                s2 = ids[index2]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = (small, large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                same_count = 0
                s1_notnone = 0
                s2_notnone = 0
                for feature in features_list:
                    if feature[index1] != '0':
                        s1_notnone += 1
                    if feature[index2] != '0':
                        s2_notnone += 1
                    if feature[index1] == feature[index2] != '0':
                        same_count += 1
                if key == 'intenso':
                    if same_count >= 4:
                        candidate_pairs.append((small, large, D[i][j]))
                    elif product_list[index1] == product_list[index2] == 'basic' and same_count >= 3:
                        candidate_pairs.append((small, large, D[i][j]))
                    else:
                        faiss_pairs.append((small, large, D[i][j]))
                elif key == 'sandisk':
                    if ('otg' in name_list[index1] and 'otg' not in name_list[index2]) or (
                            'otg' in name_list[index2] and 'otg' not in name_list[index1]):
                        continue
                    if same_count == 0 and (s1_notnone != 0 and s2_notnone != 0):
                        continue
                    if ('tesco' in name_list[index1] and 'tesco' not in name_list[index2]) or (
                            'tesco' in name_list[index2] and 'tesco' not in name_list[index1]):
                        continue
                    if item_code_list[index1] != '0' and item_code_list[index2] != '0' and item_code_list[index1] != \
                            item_code_list[index2] and mem_list[index1] == mem_list[index2] == 'sdhc':
                        continue
                    if capacity_list[index1] == capacity_list[index2] == '16g' and model_list[index1] == model_list[
                        index2] == 'ext+' and mem_list[index1] != mem_list[index2]:
                        continue
                    if series_list[index1] == series_list[index2] == 'glide' and same_count >= 3 and capacity_list[
                        index1] == capacity_list[index2] in ('256g', '512g', '1t', '2t'):
                        candidate_pairs.append((small, large, D[i][j]))
                    elif model_list[index1] == model_list[index2] == 'ext+' and same_count >= 3:
                        candidate_pairs.append((small, large, D[i][j]))
                    elif capacity_list[index1] == capacity_list[index2] == '4g' and mem_list[index1] == mem_list[
                        index2] == 'microsd':
                        candidate_pairs.append((small, large, D[i][j]))
                    elif product_list[index1] == product_list[index2] == 'otg' and same_count >= 5:
                        candidate_pairs.append((small, large, D[i][j]))
                    else:
                        faiss_pairs.append((small, large, D[i][j]))
                elif key == 'toshiba':
                    if product_list[index1] == product_list[index2] == 'xpro' and series_list[index1] == series_list[
                        index2] == 'exceria' and capacity_list[index1] == capacity_list[
                        index2] == '32g' and same_count >= 5:
                        candidate_pairs.append((small, large, D[i][j]))
                    elif model_list[index1] == model_list[index2] == 'm401':
                        candidate_pairs.append((small, large, D[i][j]))
                    elif mem_list[index1] == mem_list[index2] == 'microsd' and product_list[index1] == product_list[
                        index2] == 'x' and capacity_list[index1] == capacity_list[index2] == '64g' and same_count >= 4:
                        candidate_pairs.append((small, large, D[i][j]))
                    else:
                        faiss_pairs.append((small, large, D[i][j]))
                elif key == 'kingston':
                    if series_list[index1] == series_list[index2] == 'ultimate' and same_count >= 3:
                        candidate_pairs.append((small, large, D[i][j]))
                    elif capacity_list[index1] == capacity_list[index2] in (
                            '128g', '256g', '512g', '1t', '2t') and same_count >= 3:
                        candidate_pairs.append((small, large, D[i][j]))
                    elif product_list[index1] == product_list[index2] == 'uhs-i' and capacity_list[index1] == \
                            capacity_list[index2] == '16g' and mem_list[index1] == mem_list[
                        index2] == 'sd' and same_count >= 4:
                        candidate_pairs.append((small, large, D[i][j]))
                    elif product_list[index1] == product_list[index2] == 'flash' and same_count >= 3:
                        candidate_pairs.append((small, large, D[i][j]))
                    elif product_list[index1] == product_list[index2] == 'plus' and same_count >= 3:
                        candidate_pairs.append((small, large, D[i][j]))
                    else:
                        faiss_pairs.append((small, large, D[i][j]))
                elif key == 'pny':
                    if same_count >= 4:
                        candidate_pairs.append((small, large, D[i][j]))
                    else:
                        faiss_pairs.append((small, large, D[i][j]))
                elif key == 'lexar':
                    if mem_list[index1] == mem_list[index2] == 'xqd' and same_count >= 3:
                        candidate_pairs.append((small, large, D[i][j]))
                    else:
                        faiss_pairs.append((small, large, D[i][j]))
                else:
                    if same_count >= 5:
                        candidate_pairs.append((small, large, D[i][j]))
                    else:
                        faiss_pairs.append((small, large, D[i][j]))
        # print(key, '|', 'bucket size:', len(bucket), 'faiss pairs:', len(faiss_pairs))
        faiss_pairs.sort(key=lambda x: x[2])
        faiss_all_pairs += faiss_pairs[:int(max_limit * brand_portion[key])]

    if len(candidate_pairs) > max_limit:
        candidate_pairs.sort(key=lambda x: x[2])
        candidate_pairs = candidate_pairs[:max_limit]
    else:
        faiss_all_pairs.sort(key=lambda x: x[2])
        faiss_all_pairs = faiss_all_pairs[:max_limit - len(candidate_pairs)]
        candidate_pairs += faiss_all_pairs
    candidate_pairs = [(x[0], x[1]) for x in candidate_pairs]

    # raw_data = pd.read_csv('X2.csv')
    # gnd = pd.read_csv('Y2.csv')
    # gnd['cnt'] = 0
    # brands = ['sandisk', 'lexar', 'kingston', 'intenso', 'toshiba', 'sony', 'pny', 'samsung', 'transcend', '']
    # print('-----------------------------------------------------------------------------------------------')
    # cnt_dict: Dict[str, int] = defaultdict(int)
    # for idx in range(len(candidate_pairs)):
    #     left_id = candidate_pairs[idx][0]
    #     right_id = candidate_pairs[idx][1]
    #     index = gnd[(gnd['lid'] == left_id) & (gnd['rid'] == right_id)].index.tolist()
    #     if len(index) > 0:
    #         if len(index) > 1:
    #             raise Exception
    #         gnd['cnt'][index[0]] += 1
    #         if gnd['cnt'][index[0]] > 1:
    #             print(index)
    #     else:
    #         # wrong pairs
    #         left_sentence = raw_data[raw_data['id'] == left_id]['name'].values[0]
    #         right_sentence = raw_data[raw_data['id'] == right_id]['name'].values[0]
    #         for b in brands:
    #             if b in left_sentence.lower() or b in right_sentence.lower():
    #                 cnt_dict[b] += 1
    #                 if b == 'sandisk':
    #                     print(left_id, '|', right_id)
    #                     print(left_sentence, '|', right_sentence)
    #                 break
    #         # if left_text != right_text:
    #         #     print(idx, left_id, right_id)
    #         #     print(left_text, '|', right_text)
    #         pass
    # for key in cnt_dict.keys():
    #     cnt_dict[key] = cnt_dict[key] / gnd.shape[0]
    # print('wrong pairs:', cnt_dict)
    # print('-----------------------------------------------------------------------------------------------')
    # left = gnd[gnd['cnt'] == 0]
    # cnt_dict: Dict[str, int] = defaultdict(int)
    # for idx in left.index:
    #     # unrecognized pairs
    #     left_sentence = raw_data[raw_data['id'] == left['lid'][idx]]['name'].iloc[0]
    #     right_sentence = raw_data[raw_data['id'] == left['rid'][idx]]['name'].iloc[0]
    #     for b in brands:
    #         if b in left_sentence.lower() or b in right_sentence.lower():
    #             cnt_dict[b] += 1
    #             if b == 'sandisk':
    #                 print(left_sentence, '|', right_sentence)
    #             break
    # for key in cnt_dict.keys():
    #     cnt_dict[key] = cnt_dict[key] / gnd.shape[0]
    # print('not recognized pairs:', cnt_dict)
    # print('output pairs:\t', len(candidate_pairs))
    # print('correct pairs:\t', sum(gnd['cnt']))
    # print('recall:\t\t\t', sum(gnd['cnt']) / gnd.values.shape[0])

    return candidate_pairs
