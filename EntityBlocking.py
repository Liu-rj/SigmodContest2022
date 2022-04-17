from collections import defaultdict

import pandas as pd
from typing import *

nonsense = ['|', ',', '-', ':', '/', '+', '&']


def gen_key_sig(key, pattern1, pattern2):
    sig = 'low'
    if pattern1 != '0' and pattern2 != '0':
        key += f'.{pattern1}.{pattern2}'
        sig = 'high'
    elif pattern1 != '0':
        key += f'.{pattern1}'
        sig = 'mid'
    elif pattern2 != '0':
        key += f'.{pattern2}'
        sig = 'mid'
    return key, sig


def gen_pair_for_priority(candidates: Set, priority: Dict[str, List], max_size: int):
    if len(candidates) > max_size:
        return
    for key in priority.keys():
        bucket = priority[key]
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                if bucket[i] < bucket[j]:
                    candidates.add((bucket[i], bucket[j]))
                elif bucket[i] > bucket[j]:
                    candidates.add((bucket[j], bucket[i]))
        if len(candidates) > max_size:
            return


def block_x2(dataset: pd.DataFrame):
    """
    Give an identification for each record according to their cleaned field values
    and match records based on their identification

    param dataset: feature Dataframe of X2 after extracting

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

    sony_capacity_single = ["1tb", "256gb"]
    sony_capacity_memtype_type = ["32gb", "4gb"]

    high_priority: Dict[str, List] = defaultdict(list)
    mid_priority: Dict[str, List] = defaultdict(list)
    low_priority: Dict[str, List] = defaultdict(list)
    unidentified: List[Tuple[int, str]] = []

    for index, row in dataset.iterrows():
        instance_id = row['id']
        brand = row['brand']
        capacity = row['capacity']
        mem_type = row['mem_type']
        product_type = row['type']
        model = row['model']
        item_code = row['item_code']
        name = ' '.join(sorted(row['name'].split()))
        # name = row['name']
        series = row['series']
        pat_hb = row['pat_hb']

        high_priority[name].append(instance_id)

        if product_type == '0' and brand == "intenso" and model in model_2_type.keys():
            product_type = model_2_type[model]

        # if capacity in ('256g', '512g', '1t', '2t') and brand not in ('samsung', 'sandisk'):
        #     high_priority[f'{brand}.{capacity}'].append((instance_id, name))
        #     continue

        key = f'{brand}.{mem_type}.{capacity}.{model}.{product_type}'
        sig = 'high'
        if capacity == '0' and model == '0' and product_type == '0':
            key, sig = gen_key_sig(key, pat_hb, series)
        elif capacity == '0' and model == '0':
            key, sig = gen_key_sig(key, pat_hb, series)
        elif model == '0' and product_type == '0':
            key, sig = gen_key_sig(key, pat_hb, series)
        elif product_type == '0':
            key, sig = gen_key_sig(key, pat_hb, series)
        if brand == '0' and sig == 'mid':
            sig = 'low'
        if sig == 'high':
            high_priority[key].append(instance_id)
        elif sig == 'mid':
            mid_priority[key].append(instance_id)
        else:
            low_priority[key].append(instance_id)

    candidates = set()
    gen_pair_for_priority(candidates, high_priority, 2000000)
    gen_pair_for_priority(candidates, mid_priority, 2000000)
    gen_pair_for_priority(candidates, low_priority, 2000000)
    print('output pairs:\t', len(candidates))
    jaccard_similarities = []
    # s1 = set(bucket[i][1].split())
    # s2 = set(bucket[j][1].lower().split())
    # jaccard_similarities.append(len(s1.intersection(s2)) / max(len(s1), len(s2)))
    # candidates = [x for _, x in sorted(zip(jaccard_similarities, candidates), reverse=True)]
    return list(candidates)
