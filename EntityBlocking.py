from collections import defaultdict

import pandas as pd
from typing import *

nonsense = ['|', ',', '-', ':', '/', '+', '&']


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

    buckets: Dict[str, List] = defaultdict(list)
    unidentified: List[Tuple[int, str]] = []

    for index, row in dataset.iterrows():
        instance_id = row['id']
        brand = row['brand']
        capacity = row['capacity']
        mem_type = row['mem_type']
        product_type = row['type']
        model = row['model']
        item_code = row['item_code']
        name = row['name']
        series = row['series']

        if product_type == '0' and brand == "intenso" and model in model_2_type.keys():
            product_type = model_2_type[model]

        if capacity in ('256gb', '512gb', '1tb', '2tb') and brand not in ('samsung', 'sandisk'):
            buckets[f'{brand}.{capacity}'].append((instance_id, name))
            continue

        if brand == 'lexar':
            if capacity != '0' and product_type != '0' and mem_type != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{product_type}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))
        elif brand == 'sony':
            if (mem_type in ('ssd', 'microsd') or capacity == '1tb') and capacity != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{series}'].append((instance_id, name))
            elif mem_type != '0' and capacity != '0' and product_type != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{product_type}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))
        elif brand == 'sandisk':
            if capacity != '0' and mem_type != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{model}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))
        elif brand == 'pny':
            if capacity != '0' and mem_type != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))
        elif brand == 'intenso':
            if capacity != '0' and product_type != '0':
                buckets[f'{brand}.{capacity}.{product_type}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))
        elif brand == 'kingston':
            if mem_type != '0' and capacity != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))
        elif brand == 'samsung':
            if mem_type in ('microsd', 'ssd', 'sd', 'usb') and capacity != '0' and model != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{model}.{series}'].append((instance_id, name))
            elif mem_type != '0' and capacity != '0' and product_type != '0' and model != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{product_type}.{model}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))
        elif brand == 'toshiba':
            if capacity != '0' and mem_type != '0' and model != '0':
                buckets[f'{brand}.{capacity}.{model}.{mem_type}.{series}'].append((instance_id, name))
            elif capacity != '0' and mem_type != '0' and product_type != '0':
                buckets[f'{brand}.{capacity}.{product_type}.{mem_type}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))
        elif brand == 'transcend':
            if capacity != '0' and mem_type != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))
        else:
            if brand != '0' and capacity != '0' and mem_type != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{series}'].append((instance_id, name))
            else:
                unidentified.append((instance_id, name))

    # solved_classes = set()
    # for s in solved_spec:
    #     if s['capacity'] != '0' and s['mem_type'] != '0':
    #         solved_classes.add(s['brand'] + s['capacity'] + s['mem_type'])
    # unsolved_spec_cp = unsolved_spec.copy()
    # for u in unsolved_spec_cp:
    #     if u['capacity'] != '0' and u['mem_type'] != '0' and (
    #             u['type'] != '0' or u['model'] != '0'):
    #         if (u['brand'] + u['capacity'] +
    #             u['mem_type']) not in solved_classes:
    #             u['identification'] = u['brand'] + u['capacity'] + \
    #                                   u['mem_type'] + u['type'] + u['model']
    #             solved_spec.append(u)
    #             unsolved_spec.remove(u)
    #             solved_classes.add(u['brand'] + u['capacity'] + u['mem_type'])
    # unsolved_spec_cp = unsolved_spec.copy()
    # for u in unsolved_spec_cp:
    #     if u['capacity'] != '0' and u['mem_type'] != '0':
    #         if (u['brand'] + u['capacity'] +
    #             u['mem_type']) not in solved_classes:
    #             u['identification'] = u['brand'] + u['capacity'] + \
    #                                   u['mem_type'] + u['type'] + u['model']
    #             solved_spec.append(u)
    #             unsolved_spec.remove(u)
    #             solved_classes.add(u['brand'] + u['capacity'] + u['mem_type'])
    # unsolved_spec_cp = unsolved_spec.copy()
    # for u in unsolved_spec_cp:
    #     if u['capacity'] != '0':
    #         if (u['brand'] + u['capacity'] +
    #             u['mem_type']) not in solved_classes:
    #             u['identification'] = u['brand'] + u['capacity'] + \
    #                                   u['mem_type'] + u['type'] + u['model']
    #             solved_spec.append(u)
    #             unsolved_spec.remove(u)
    #             solved_classes.add(u['brand'] + u['capacity'] + u['mem_type'])
    #
    # unsolved_spec_cp = unsolved_spec.copy()
    # solved_spec_cp = solved_spec.copy()
    #
    # for u in unsolved_spec_cp:
    #     for s in solved_spec_cp:
    #         if u['item_code'] != '0' and u['item_code'] == s['item_code']:
    #             u['identification'] = s['identification']
    #             solved_spec.append(u)
    #             if u in unsolved_spec:
    #                 unsolved_spec.remove(u)
    #
    # unsolved_spec_cp = unsolved_spec.copy()
    # solved_spec_cp = solved_spec.copy()
    #
    # for u in unsolved_spec_cp:
    #     if u['brand'] == 'sandisk':
    #         continue
    #     for s in solved_spec_cp:
    #         if u['brand'] != '0' and u['capacity'] != '0' and u['mem_type'] != '0' and u['type'] != '0':
    #             if u['brand'] == s['brand'] and u['capacity'] == s['capacity'] and u['mem_type'] == s['mem_type'] and \
    #                     u['type'] == s['type']:
    #                 u['identification'] = s['identification']
    #                 solved_spec.append(u)
    #                 unsolved_spec.remove(u)
    #                 break
    #         elif u['brand'] != '0' and u['capacity'] != '0' and u['mem_type'] != '0' and u['model'] != '0':
    #             if u['brand'] == s['brand'] and u['capacity'] == s['capacity'] and u['mem_type'] == s['mem_type'] and \
    #                     u['model'] == s['model']:
    #                 u['identification'] = s['identification']
    #                 solved_spec.append(u)
    #                 unsolved_spec.remove(u)
    #                 break
    #         elif u['brand'] != '0' and u['capacity'] != '0' and u['type'] != '0' and u['model'] != '0':
    #             if u['brand'] == s['brand'] and u['capacity'] == s['capacity'] and u['type'] == s['type'] and \
    #                     u['model'] == s['model']:
    #                 u['identification'] = s['identification']
    #                 solved_spec.append(u)
    #                 unsolved_spec.remove(u)
    #                 break
    #         elif u['brand'] != '0' and u['capacity'] != '0' and u['mem_type'] != '0':
    #             if u['brand'] == s['brand'] and u['capacity'] == s['capacity'] and u['mem_type'] == s['mem_type']:
    #                 u['identification'] = s['identification']
    #                 solved_spec.append(u)
    #                 unsolved_spec.remove(u)
    #                 break
    #         elif u['brand'] != '0' and u['capacity'] != '0' and u['type'] != '0':
    #             if u['brand'] == s['brand'] and u['capacity'] == s['capacity'] and u['type'] == s['type']:
    #                 u['identification'] = s['identification']
    #                 solved_spec.append(u)
    #                 unsolved_spec.remove(u)
    #                 break
    #
    # clusters = dict()
    #
    # for s in solved_spec:
    #     if s['identification'] in clusters.keys():
    #         clusters[s['identification']].append(s['id'])
    #     else:
    #         clusters.update({s['identification']: [s['id']]})
    #
    # for u in unsolved_spec:
    #     if u['title'] in clusters.keys():
    #         clusters[u['title']].append(u['id'])
    #     else:
    #         clusters.update({u['title']: [u['id']]})

    # print('unidentified size: ', len(unidentified))

    candidates = []
    for key in buckets.keys():
        bucket = buckets[key]
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                if bucket[i][0] < bucket[j][0]:
                    candidates.append((bucket[i][0], bucket[j][0]))
                elif bucket[i][0] > bucket[j][0]:
                    candidates.append((bucket[j][0], bucket[i][0]))
    return candidates
