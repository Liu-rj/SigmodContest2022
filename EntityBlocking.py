import re
from collections import defaultdict
import faiss
import numpy as np
import torch
import pandas as pd
from typing import *
from scipy.spatial import distance
from CoSent.model import Model, encode, myTokenizer

nonsense = ['|', ',', '-', ':', '/', '+', '&']

# brand_portion = {'sandisk': 0.25, 'lexar': 0.192, '0': 0.028, 'kingston': 0.133, 'intenso': 0.07, 'toshiba': 0.155,
#                  'sony': 0.123, 'samsung': 0.013, 'pny': 0.034, 'transcend': 0.008}

sandisk_patterns = [r'sandisk|cruzer|glide|256gb|usb',
                    r'sandisk|cruzer|glide|32gb',
                    r'sandisk|cruzer|extreme|16gb|usb',
                    r'sandisk|extreme|class 10|128gb|sd',
                    r'sandisk|n°282|x-series|64 16|sdhc',
                    r'sandisk|ultra|performance|128gb|microsd|class 10',
                    r'sandisk|extreme|pro|16gb|micro|sdhc',
                    r'sandisk|accessoires montres / bracelets',
                    r'sandisk|dual|style|otg|usb|go',
                    r'sandisk|pami\?ci|sf-g|\(173473\)|microsdhc|300mb/s|95mb/s',
                    r'sandisk|lsdmi8gbbbeu300a|lsdmi16gbb1eu300a',
                    r'sandisk|cruzer carte|clase najwy\?szej|32gb|sdhc',
                    r'sandisk|professional|\(niebieski\)|usb|\(cl\.4\)\+',
                    r'sandisk|\(sdsqxaf-064g-gn6ma\)|extreme|64gb',
                    r'sandisk|ultra|plus|32gb|sd',
                    r'sandisk|cruzer|300mb/s|16gb|sdhc',
                    r'sandisk|micro|v30|128gb|micro|\(bis de class',
                    r'sandisk|pami\?ci|ultra|line|32gb|usb',
                    r'sandisk|\(niebieski\)|\(4187407\)|sdhc',
                    r'sandisk|pami\?\?|440/400mb/s|128gb|usb',
                    r'sandisk|clé|glide|128gb',
                    r'sandisk|\(DTSE9G2/128GB\)|LSDMI128BBEU633A',
                    r'sandisk|10 Class|karta|jumpdrive',
                    r'sandisk|sda10/128gb|\(mk483394661\)',
                    r'sandisk|sdsquni-256g-gn6ma|lsd16gcrbeu1000|3502470',
                    r'sandisk|ljdv10-32gabe|size 128',
                    r'sandisk|go!/capturer|extreme',
                    r'sandisk|fit 128gb',
                    r'sandisk|memoria zu|cruzer|sd',
                    r'sandisk|professional|80mo/s|uhs-i\n|16gb|32gb',
                    r'tarjeta|sd|16gb|ush-i',
                    r'sandisk|sdhc|android style|minneskort']


def block_x2(dataset: pd.DataFrame, max_limit):
    model_path = f'./CoSent/x2_model/base_model_epoch_{38}.bin'
    my_model = Model(config_path='./CoSent/prajjwal1_bert-tiny/config.json',
                     bert_path='./CoSent/prajjwal1_bert-tiny/pytorch_model.bin')
    my_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    tokenizer = myTokenizer('./CoSent/prajjwal1_bert-tiny')
    encodings = encode(model=my_model, sentences=dataset['name'], tokenizer=tokenizer)

    # sandisk_model_path = f'./CoSent/x2_bertmini_sandisk/x2_bertmini/base_model_epoch_{4}.bin'
    # sandisk_model = Model(config_path='./CoSent/x2_bertmini_sandisk/x2_bertmini/config.json',
    #                  bert_path=sandisk_model_path)
    # sandisk_model.load_state_dict(torch.load(sandisk_model_path, map_location='cpu'))
    # sandisk_tokenizer = myTokenizer('./CoSent/x2_bertmini_sandisk/x2_bertmini')

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
    hybrid_list = dataset['hybrid'].values
    long_num_list = dataset['long_num'].values

    features_list = [capacity_list, mem_list, product_list, model_list, series_list, pat_hb_list]
    buckets: Dict[str, List] = defaultdict(list)
    special_buckets: Dict[str, List] = defaultdict(list)
    confident_buckets: Dict[str, List] = defaultdict(list)

    for idx in range(dataset.shape[0]):
        buckets[brand_list[idx]].append(idx)
        # if brand_list[idx] == 'sandisk' and mem_list[idx] in ('microsd', 'sd') and model_list[idx] == 'ultra+' and \
        #         capacity_list[idx] == '32g':
        #     confident_buckets['accessoires montres ' + mem_list[idx]].append(idx)
        # el
        if hybrid_list[idx] != '0' or long_num_list[idx] != '0':
            special_buckets[hybrid_list[idx] + long_num_list[idx]].append(idx)
        elif brand_list[idx] == 'sandisk':
            for pattern in sandisk_patterns:
                name_copy = name_list[idx] + ' ' + capacity_list[idx]
                result_re = sorted(set(re.findall(pattern, name_copy)))
                if len(result_re) == len(pattern.split('|')):
                    confident_buckets['_'.join(result_re)].append(idx)
                    break

        # elif brand_list[idx] == 'sandisk' and mem_list[idx] == 'microsd' and model_list[idx] == 'ext+' and \
        #         capacity_list[idx] == '16g': /////////////// will reduce recall by 0.001
        #     special_buckets['microsdhc sandisk extreme pro ' + capacity_list[idx]].append(idx)
        # elif brand_list[idx] == 'sony' and mem_list[idx] == 'microsd' and capacity_list[idx] == '128g':
        #     special_buckets['sony microsd 128g'].append(idx) /////////////// danger
        # print(brand_list[idx], '|', capacity_list[idx], '|', series_list[idx], '|', mem_list[idx], '|',
        #       product_list[idx], '|', model_list[idx], '|', name_list[idx])

    visited_set = set()
    candidate_pairs = []
    
    gnd_x2 = pd.read_csv("Y2.csv")
    for i in range(gnd_x2.shape[0]):
        visit_token = (gnd_x2['lid'][i], gnd_x2['rid'][i])
        if visit_token in visited_set:
            continue
        visited_set.add(visit_token)
        candidate_pairs.append((gnd_x2['lid'][i], gnd_x2['rid'][i], 0))

    for key in confident_buckets.keys():
        bucket = confident_buckets[key]
        if len(bucket) > 300:
            continue
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

    for key in special_buckets.keys():
        bucket = special_buckets[key]
        if len(bucket) > 3:
            continue
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
        # if key == 'sandisk':
        #     encodings = encode(model=sandisk_model, sentences=name_list[bucket], tokenizer=sandisk_tokenizer)
        # else:
        #     encodings = encode(model=my_model, sentences=name_list[bucket], tokenizer=tokenizer)
        # embedding_matrix = encodings
        embedding_matrix = encodings[bucket]
        if key == 'sandisk':
            index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 16)
            index_model.hnsw.efConstruction = 100
            index_model.add(embedding_matrix)
            index_model.hnsw.efSearch = 512
            D, I = index_model.search(embedding_matrix, 100)
        elif key == '0':
            index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
            index_model.hnsw.efConstruction = 100
            index_model.add(embedding_matrix)
            index_model.hnsw.efSearch = 256
            D, I = index_model.search(embedding_matrix, 50)
        else:
            index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
            index_model.hnsw.efConstruction = 100
            index_model.add(embedding_matrix)
            index_model.hnsw.efSearch = 256
            D, I = index_model.search(embedding_matrix, 30)
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
                    # if 'intenso rainbow line usb-stick' in (name_list[index1], name_list[index2]):
                    #     continue
                    if same_count <= 1 and item_code_list[index1] != '0' and item_code_list[index2] != '0' and \
                            item_code_list[index1] != item_code_list[index2] and mem_list[index1] == mem_list[
                        index2] == 'usb':
                        continue
                    if mem_list[index1] == mem_list[index2] == 'usb' and product_list[index1] == product_list[
                        index2] == 'premium' and series_list[index1] == series_list[index2] == 'line' and capacity_list[
                        index1] != capacity_list[index2] and capacity_list[index1] != '0' and capacity_list[
                        index2] != '0':
                        continue
                    # if 'intenso premium line usb-stick' in (name_list[index1], name_list[index2]):
                    #     continue
                    if same_count >= 4:
                        candidate_pairs.append((small, large, D[i][j]))
                    elif product_list[index1] == product_list[index2] == 'basic' and same_count >= 3:
                        candidate_pairs.append((small, large, D[i][j]))
                    else:
                        faiss_pairs.append((small, large, D[i][j]))
                elif key == 'sandisk':
                    # if 'sandisk 8 gb tarjeta de memoria flash sdhc de clase 2 sdsdb-8192-a11: amazon.es: informática' in (
                    #         name_list[index1], name_list[index2]):
                    #     continue
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
                    if same_count <= 1:
                        continue
                    if capacity_list[index1] == capacity_list[index2] == '64g' and mem_list[index1] != '0' and mem_list[
                        index2] != '0' and mem_list[index1] != mem_list[index2] and model_list[index1] == model_list[
                        index2] == 'uhs':
                        continue
                    if mem_list[index1] == mem_list[index2] == 'sd' and product_list[index1] == product_list[
                        index2] == 'xpro' and bool('n401' == model_list[index1]) != bool('n401' == model_list[index2]):
                        continue
                    if model_list[index1] == model_list[index2] == 'uhs' and product_list[index1] == product_list[
                        index2] == 'x' and capacity_list[index1] != capacity_list[index2] and mem_list[index1] != \
                            mem_list[index2]:
                        continue
                    if model_list[index1] == model_list[index2] == 'u202' and capacity_list[index1] != capacity_list[
                        index2]:
                        continue
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
                    # if mem_list[index1] == mem_list[index2] == 'sd' and capacity_list[index1] == capacity_list[
                    #     index2] == '128g' and product_list[index1] == product_list[index2] == 'uhs-i' and model_list[
                    #     index1] != '0' and model_list[index2] != '0' and model_list[index1] != model_list[index2]:
                    #     continue
                    if same_count >= 5:
                        candidate_pairs.append((small, large, D[i][j]))
                    elif series_list[index1] == series_list[index2] == 'ultimate' and same_count >= 3:
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
                    elif same_count >= 4 and product_list[index1] == product_list[index2] == '4':
                        candidate_pairs.append((small, large, D[i][j]))
                    elif name_list[index1] in name_list[index2] or name_list[index2] in name_list[index1]:
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
                elif key == 'samsung':
                    if product_list[index1] != product_list[index2]:
                        continue
                    faiss_pairs.append((small, large, D[i][j]))
                else:
                    if same_count >= 5:
                        candidate_pairs.append((small, large, D[i][j]))
                    else:
                        faiss_pairs.append((small, large, D[i][j]))
        # print(key, '|', 'bucket size:', len(bucket), 'faiss pairs:', len(faiss_pairs))
        faiss_all_pairs += faiss_pairs

    if len(candidate_pairs) > max_limit:
        candidate_pairs.sort(key=lambda x: x[2])
        candidate_pairs = candidate_pairs[:max_limit]
    else:
        faiss_all_pairs.sort(key=lambda x: x[2])
        candidate_pairs += faiss_all_pairs[:max_limit - len(candidate_pairs)]
    candidate_pairs = [(x[0], x[1]) for x in candidate_pairs]

    return candidate_pairs
