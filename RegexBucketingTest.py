from collections import defaultdict
import faiss
import numpy as np
import torch
import pandas as pd
from typing import *
from scipy.spatial import distance
from CoSent.model import Model, encode
import pandas as pd
from FeatureExtracting import extract_x2
from transformers.models.bert import BertTokenizer

nonsense = ['|', ',', '-', ':', '/', '+', '&']


def block_x2(dataset: pd.DataFrame, max_limit):
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

    model_path = f'./CoSent/x2_model/base_model_epoch_{38}.bin'
    my_model = Model(config_path='./CoSent/prajjwal1_bert-tiny/config.json',
                     bert_path='./CoSent/prajjwal1_bert-tiny/pytorch_model.bin')
    my_model.load_state_dict(torch.load(model_path, map_location='cpu'))

    sony_capacity_single = ["1tb", "256gb"]
    sony_capacity_memtype_type = ["32gb", "4gb"]

    same_name: Dict[str, List] = defaultdict(list)
    all_buckets: Dict[str, Tuple[List, Dict]] = {}

    ids = dataset['id'].values
    series_list = dataset['series'].values
    pat_hb_list = dataset['pat_hb'].values
    brand_list = dataset['brand'].values
    capacity_list = dataset['capacity'].values
    mem_list = dataset['mem_type'].values
    product_list = dataset['type'].values
    model_list = dataset['model'].values
    name_list = dataset['name'].values
    for idx in range(dataset.shape[0]):
        brand = brand_list[idx]
        capacity = capacity_list[idx]
        mem_type = mem_list[idx]
        product_type = product_list[idx]
        model = model_list[idx]
        series = series_list[idx]
        # item_code = dataset['item_code'][idx]
        name = ''.join(sorted(name_list[idx].split()))

        same_name[name].append(idx)

        outer_key = brand
        if outer_key not in all_buckets.keys():
            buckets: Dict[str, List] = defaultdict(list)
            all_buckets[outer_key] = ([], buckets)
        buckets = all_buckets[outer_key][1]
        unidentified = all_buckets[outer_key][0]

        if product_type == '0' and brand == "intenso" and model in model_2_type.keys():
            product_type = model_2_type[model]

        # if brand == 'sony':
        #     print(series, '|', mem_type, '|', product_type, '|', model, '|', name_list[idx])

        if capacity in ('256g', '512g', '1t', '2t'):
            if brand in ('intenso', 'pny', 'samsung', 'tesco', 'toshiba'):
                buckets[f'{brand}.{capacity}'].append(idx)
            elif brand == 'sandisk':
                buckets[f'{brand}.{capacity}.{mem_type}.{model}'].append(idx)
            elif brand == 'kingston':
                buckets[f'{brand}.{capacity}.{mem_type}'].append(idx)
            elif brand == 'sony':
                buckets[f'{brand}.{capacity}.{mem_type}.{product_type}'].append(idx)
            elif brand == 'lexar':
                buckets[f'{brand}.{capacity}.{product_type}.{mem_type}'].append(idx)
            else:
                buckets[f'{brand}.{capacity}.{series}'].append(idx)
            continue

        if brand == 'lexar':
            if capacity != '0' and product_type != '0' and mem_type != '0':
                key = brand + capacity + mem_type + product_type
                buckets[key].append(idx)
            else:
                unidentified.append(idx)
        elif brand == 'sony':
            if (mem_type in ('ssd', 'microsd') or capacity == '1tb') and capacity != '0':
                key = brand + capacity + mem_type
                buckets[key].append(idx)
            elif mem_type != '0' and capacity != '0' and product_type != '0':
                key = brand + capacity + mem_type + product_type
                buckets[key].append(idx)
            else:
                unidentified.append(idx)
        elif brand == 'sandisk':
            if capacity != '0' and mem_type != '0':
                key = brand + capacity + mem_type + model
                buckets[key].append(idx)
            else:
                unidentified.append(idx)
        elif brand == 'pny':
            if capacity != '0' and mem_type != '0':
                key = brand + capacity + mem_type
                buckets[key].append(idx)
            else:
                unidentified.append(idx)
        elif brand == 'intenso':
            if capacity != '0' and product_type != '0':
                key = brand + capacity + product_type
                buckets[key].append(idx)
            else:
                unidentified.append(idx)
        elif brand == 'kingston':
            if mem_type != '0' and capacity != '0':
                key = brand + capacity + mem_type
                buckets[key].append(idx)
            else:
                unidentified.append(idx)
        elif brand == 'samsung':
            if mem_type in ('microsd', 'ssd', 'sd', 'usb') and capacity != '0' and model != '0':
                key = brand + capacity + mem_type + model
                buckets[key].append(idx)
            elif mem_type != '0' and capacity != '0' and product_type != '0' and model != '0':
                key = brand + capacity + mem_type + product_type + model
                buckets[key].append(idx)
            else:
                unidentified.append(idx)
        elif brand == 'toshiba':
            if capacity != '0' and mem_type != '0' and model != '0':
                key = brand + capacity + model + mem_type
                buckets[key].append(idx)
            elif capacity != '0' and mem_type != '0' and product_type != '0':
                key = brand + capacity + product_type + mem_type
                buckets[key].append(idx)
            else:
                unidentified.append(idx)
        elif brand == 'transcend':
            if capacity != '0' and mem_type != '0':
                key = brand + capacity + mem_type
                buckets[key].append(idx)
            else:
                unidentified.append(idx)
        else:
            key = brand + capacity + mem_type + series
            buckets[key].append(idx)

    visited_set = set()
    candidates = []
    for key in same_name.keys():
        bucket = same_name[key]
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                s1 = ids[bucket[i]]
                s2 = ids[bucket[j]]
                small = min(s1, s2)
                large = max(s1, s2)
                symbol = str(small) + ' ' + str(large)
                if symbol not in visited_set:
                    visited_set.add(symbol)
                    candidates.append((small, large))

    large_pairs = []
    tokenizer = BertTokenizer.from_pretrained('./CoSent/x2_model')
    for outer_key in all_buckets.keys():
        unidentified = all_buckets[outer_key][0]
        uid_encodings = encode(model=my_model, sentences=name_list[unidentified], tokenizer=tokenizer)
        buckets = all_buckets[outer_key][1]
        for key in buckets.keys():
            bucket = buckets[key]
            # bucket = bucket + unidentified
            if len(bucket) < 0:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        s1 = ids[bucket[i]]
                        s2 = ids[bucket[j]]
                        small = min(s1, s2)
                        large = max(s1, s2)
                        symbol = str(small) + ' ' + str(large)
                        if symbol not in visited_set:
                            visited_set.add(symbol)
                            candidates.append((small, large))
            else:
                encodings = encode(model=my_model, sentences=name_list[bucket], tokenizer=tokenizer)
                if len(unidentified) > 0:
                    new_buck = unidentified + bucket
                    embedding_matrix = np.concatenate((uid_encodings, encodings), axis=0)
                else:
                    new_buck = bucket
                    embedding_matrix = encodings
                index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
                index_model.hnsw.efConstruction = 100
                index_model.add(embedding_matrix)
                index_model.hnsw.efSearch = 256
                D, I = index_model.search(embedding_matrix, 50)
                for i in range(len(D)):
                    for j in range(len(D[0])):
                        s1 = ids[new_buck[i]]
                        s2 = ids[new_buck[I[i][j]]]
                        if s1 == s2:
                            continue
                        small = min(s1, s2)
                        large = max(s1, s2)
                        visit_token = str(small) + ' ' + str(large)
                        if visit_token in visited_set:
                            continue
                        visited_set.add(visit_token)
                        large_pairs.append((small, large, D[i][j]))
    large_pairs.sort(key=lambda x: x[2])
    large_pairs = large_pairs[:max_limit - len(candidates)]
    large_pairs = [(x[0], x[1]) for x in large_pairs]
    return candidates + large_pairs


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


if __name__ == '__main__':
    raw_data = pd.read_csv('X1.csv')
    raw_data = pd.read_csv('X2.csv')
    raw_data['name'] = raw_data.name.str.lower()
    features = extract_x2(raw_data)
    candidates_x2 = block_x2(features, 2000000)
    candidates_x1 = []
    # save_output(candidates_x1, 1000000, candidates_x2, 2000000)

    gnd = pd.read_csv('Y2.csv')
    gnd['cnt'] = 0
    for idx in range(len(candidates_x2)):
        left_id = candidates_x2[idx][0]
        right_id = candidates_x2[idx][1]
        index = gnd[(gnd['lid'] == left_id) & (gnd['rid'] == right_id)].index.tolist()
        if len(index) > 0:
            if len(index) > 1:
                raise Exception
            gnd['cnt'][index[0]] += 1
            if gnd['cnt'][index[0]] > 1:
                print(index)
        else:
            # left_text = raw_data[raw_data['id'] == left_id]['name'].values[0]
            # right_text = raw_data[raw_data['id'] == right_id]['name'].values[0]
            # if left_text != right_text:
            #     print(idx, left_id, right_id)
            #     print(left_text, '|', right_text)
            pass
    print('output pairs:\t', len(candidates_x2))
    print('correct pairs:\t', sum(gnd['cnt']))
    print('recall:\t\t\t', sum(gnd['cnt']) / gnd.values.shape[0])
    # left = gnd[gnd['cnt'] == 0].reset_index()
    # for idx in range(left.shape[0]):
    #     print(left['lid'][idx], ',', left['rid'][idx])
    #     print(raw_data[raw_data['id'] == left['lid'][idx]]['name'].iloc[0])
    #     print(raw_data[raw_data['id'] == left['rid'][idx]]['name'].iloc[0])
    #     print()
