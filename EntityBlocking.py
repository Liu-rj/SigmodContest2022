from collections import defaultdict

import pandas as pd
from typing import *

def block_x1(dataset: pd.DataFrame):
    """ Call clean_x2.py;

    Give an identification for each record according to their cleaned field values
    and match records based on their identification

    param dataset: feature Dataframe of X1 after extracting

    :return:
            A DataFrame of matched pairs which contains following columns:
            {left_instance_id: the left instance of a matched pair
             left_instance_id: the right instance of a matched pair}
    """
    pc_aliases = {
        "2320": "3435", "v7482": "v7582", "810g2": "810", "2338": "2339",
        "346058u": "3460", "4291": "4290", "4287": "4290", "0622": "0627"}

    cpu_model_aliases = {
        "hp": {"2410m": "2540m", "2620m": "2640m"},
        "acer": {},
        "lenovo": {},
        "asus": {},
        "dell": {}
    }

    model_family_2_pcname = {
        "4010u aspire": "e1572"
    }

    pc_single = ["v7582", "15f009wm", "3093", "ux31a", "v3772", "v3572", "m731r", "e3111",
                 "v3111", "15p030nr", "e5771", "e1731", "3437 ", "2170p", "e1532", "e1522",
                 "e1571", "e5571", "15d053cl", "v5573", "3448", "8460p", "8570p",
                 "2570p", "2760p", "0596", "547578", "547150", "547375"]

    pc_core = ["e1572", "e1771", "810", "8560p", "3438"]
    pc_model_capacity = ["2325", "3460"]
    pc_capacity = ["9470m", "3444", "2339"]

    solved_spec = []
    unsolved_spec = []
    instance_list = set()

    for index, row in dataset.iterrows():
        instance_id = row['id']
        brand = row['brand']
        cpu_core = row['cpu_core']
        cpu_model = row['cpu_model']
        cpu_frequency = row['cpu_frequency']
        display_size = row['display_size']
        pc_name = row['pc_name']
        capacity = row['ram_capacity']
        family = row['family']
        title = row['title']
        pc = {}

        if (cpu_model + ' ' + family) in model_family_2_pcname.keys():
            pc_name = model_family_2_pcname[(cpu_model + ' ' + family)]

        if pc_name in pc_aliases.keys():
            pc_name = pc_aliases[pc_name]

        if brand in cpu_model_aliases.keys():
            if cpu_model in cpu_model_aliases[brand].keys():
                cpu_model = cpu_model_aliases[brand][cpu_model]

        instance_list.add(instance_id)

        pc['id'] = instance_id
        pc['title'] = title
        pc['brand'] = brand
        pc['pc_name'] = pc_name
        pc['cpu_model'] = cpu_model
        pc['capacity'] = capacity
        pc['cpu_core'] = cpu_core
        pc['family'] = family
        pc['cpu_frequency'] = cpu_frequency
        pc['display_size'] = display_size

        if pc_name != '0' and cpu_model != '0' and capacity != '0' and cpu_core != '0':
            pc['identification'] = brand + ' ' + pc_name + \
                                   ' ' + cpu_model + ' ' + capacity + ' ' + cpu_core
            solved_spec.append(pc)
        else:
            unsolved_spec.append(pc)

    for u in unsolved_spec.copy():
        for s in solved_spec.copy():
            if u['brand'] != '0' and u['pc_name'] != '0' and u['capacity'] != '0' and u['cpu_model'] != '0':
                if u['brand'] == s['brand'] and u['pc_name'] == s['pc_name'] and u['capacity'] == s['capacity'] and \
                        u['cpu_model'] == s['cpu_model']:
                    if u['family'] == '0' or s['family'] == '0' or u['family'] == s['family']:
                        u['identification'] = s['identification']
                        solved_spec.append(u)
                        unsolved_spec.remove(u)
                        break
            elif u['brand'] != '0' and u['pc_name'] != '0' and u['cpu_core'] != '0' and u['cpu_model'] != '0':
                if u['brand'] == s['brand'] and u['pc_name'] == s['pc_name'] and u['cpu_model'] == s['cpu_model'] and \
                        u['cpu_core'] == s['cpu_core']:
                    if u['family'] == '0' or s['family'] == '0' or u['family'] == s['family']:
                        u['identification'] = s['identification']
                        solved_spec.append(u)
                        unsolved_spec.remove(u)
                        break
            elif u['brand'] != '0' and u['capacity'] != '0' and u['cpu_core'] != '0' and u['pc_name'] != '0':
                if u['brand'] == s['brand'] and u['pc_name'] == s['pc_name'] and u['cpu_core'] == s['cpu_core'] and \
                        u['capacity'] == s['capacity']:
                    if u['family'] == '0' or s['family'] == '0' or u['family'] == s['family']:
                        u['identification'] = s['identification']
                        solved_spec.append(u)
                        unsolved_spec.remove(u)
                        break
            elif u['brand'] != '0' and u['capacity'] != '0' and u['cpu_core'] != '0' and u['cpu_model'] != '0':
                if u['brand'] == s['brand'] and u['capacity'] == s['capacity'] and u['cpu_core'] == s['cpu_core'] and \
                        u['cpu_model'] == s['cpu_model']:
                    if u['family'] == '0' or s['family'] == '0' or u['family'] == s['family']:
                        u['identification'] = s['identification']
                        solved_spec.append(u)
                        unsolved_spec.remove(u)
                        break
            elif u['brand'] != '0' and u['cpu_model'] != '0' and u['pc_name'] != '0':
                if u['brand'] == s['brand'] and u['pc_name'] == s['pc_name'] and u['cpu_model'] == s['cpu_model']:
                    if u['family'] == '0' or s['family'] == '0' or u['family'] == s['family']:
                        u['identification'] = s['identification']
                        solved_spec.append(u)
                        unsolved_spec.remove(u)
                        break
            elif u['brand'] != '0' and u['capacity'] != '0' and u['cpu_model'] != '0' and u['display_size'] != '0' and \
                    u['cpu_frequency'] != '0':
                if u['brand'] == s['brand'] and u['capacity'] == s['capacity'] and \
                        u['cpu_model'] == s['cpu_model'] and u['display_size'] == s['display_size'] and \
                        u['cpu_frequency'] == s['cpu_frequency']:
                    if u['family'] == '0' or s['family'] == '0' or u['family'] == s['family']:
                        u['identification'] = s['identification']
                        solved_spec.append(u)
                        unsolved_spec.remove(u)
                        break
            elif u['brand'] != '0' and u['capacity'] != '0' and u['pc_name'] != '0' and u['display_size'] != '0' and \
                    u['cpu_frequency'] != '0':
                if u['brand'] == s['brand'] and u['capacity'] == s['capacity'] and u['pc_name'] == s['pc_name'] and \
                        u['display_size'] == s['display_size'] and u['cpu_frequency'] == s['cpu_frequency']:
                    if u['family'] == '0' or s['family'] == '0' or u['family'] == s['family']:
                        u['identification'] = s['identification']
                        solved_spec.append(u)
                        unsolved_spec.remove(u)
                        break

    for i in unsolved_spec:
        if i in solved_spec:
            continue
        for j in unsolved_spec:
            if j in solved_spec:
                continue
            if i['id'] == j['id']:
                continue
            if i['brand'] == j['brand'] and i['capacity'] == j['capacity'] and \
                    i['cpu_core'] == j['cpu_core'] and i['cpu_model'] == j['cpu_model'] and \
                    i['pc_name'] == j['pc_name']:
                i['identification'] = i['brand'] + i['capacity'] + \
                                      i['cpu_core'] + i['cpu_model'] + i['pc_name']
                j['identification'] = i['identification']
                if i not in solved_spec:
                    solved_spec.append(i)
                if j not in solved_spec:
                    solved_spec.append(j)

    clusters = dict()

    for s in solved_spec:
        if s['identification'] in clusters.keys():
            clusters[s['identification']].append(s['id'])
        else:
            clusters.update({s['identification']: [s['id']]})

    for u in unsolved_spec:
        if u['title'] in clusters.keys():
            clusters[u['title']].append(u['id'])
        else:
            clusters.update({u['title']: [u['id']]})

    couples = set()
    for c in clusters.keys():
        if len(clusters[c]) > 1:
            for i in clusters[c]:
                for j in clusters[c]:
                    if i < j:
                        couples.add((i, j))
                    if i > j:
                        couples.add((j, i))

    return list(couples)


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
        title = row['name']

        if product_type == '0' and brand == "intenso" and model in model_2_type.keys():
            product_type = model_2_type[model]

        if capacity in ('256gb', '512gb', '1tb', '2tb') and brand not in ('samsung', 'sandisk'):
            buckets[f'{brand}.{capacity}'].append(instance_id)
            continue

        # if brand == 'lexar':
        #     if capacity != '0' and product_type != '0' and mem_type != '0':
        #         buckets[f'{brand}.{capacity}.{mem_type}.{product_type}'].append(instance_id)
        #     else:
        #         unidentified.append((instance_id, title))
        # elif brand == 'sony':
        #     if (mem_type in ('ssd', 'microsd') or capacity == '1tb') and capacity != '0':
        #         buckets[f'{brand}.{capacity}.{mem_type}'].append(instance_id)
        #     elif mem_type != '0' and capacity != '0' and product_type != '0':
        #         buckets[f'{brand}.{capacity}.{mem_type}.{product_type}'].append(instance_id)
        #     else:
        #         unidentified.append((instance_id, title))
        # el
        if brand == 'sandisk':
            if capacity != '0' and mem_type != '0':
                buckets[f'{brand}.{capacity}.{mem_type}.{model}'].append(instance_id)
            else:
                unidentified.append((instance_id, title))
        # elif brand == 'pny':
        #     if capacity != '0' and mem_type != '0':
        #         buckets[f'{brand}.{capacity}.{mem_type}'].append(instance_id)
        #     else:
        #         unidentified.append((instance_id, title))
        # elif brand == 'intenso':
        #     if capacity != '0' and product_type != '0':
        #         buckets[f'{brand}.{capacity}.{product_type}'].append(instance_id)
        #     else:
        #         unidentified.append((instance_id, title))
        # elif brand == 'kingston':
        #     if mem_type != '0' and capacity != '0':
        #         buckets[f'{brand}.{capacity}.{mem_type}'].append(instance_id)
        #     else:
        #         unidentified.append((instance_id, title))
        # elif brand == 'samsung':
        #     if mem_type in ('microsd', 'ssd', 'sd', 'usb') and capacity != '0' and model != '0':
        #         buckets[f'{brand}.{capacity}.{mem_type}.{model}'].append(instance_id)
        #     elif mem_type != '0' and capacity != '0' and product_type != '0' and model != '0':
        #         buckets[f'{brand}.{capacity}.{mem_type}.{product_type}.{model}'].append(instance_id)
        #     else:
        #         unidentified.append((instance_id, title))
        # elif brand == 'toshiba':
        #     if capacity != '0' and mem_type != '0' and model != '0':
        #         buckets[f'{brand}.{capacity}.{model}.{mem_type}'].append(instance_id)
        #     elif capacity != '0' and mem_type != '0' and product_type != '0':
        #         buckets[f'{brand}.{capacity}.{product_type}.{mem_type}'].append(instance_id)
        #     else:
        #         unidentified.append((instance_id, title))
        # elif brand == 'transcend':
        #     if capacity != '0' and mem_type != '0':
        #         buckets[f'{brand}.{capacity}.{mem_type}'].append(instance_id)
        #     else:
        #         unidentified.append((instance_id, title))
        # else:
        #     if brand != '0' and capacity != '0' and mem_type != '0':
        #         buckets[f'{brand}.{capacity}.{mem_type}'].append(instance_id)
        #     else:
        #         unidentified.append((instance_id, title))

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

    print('unidentified size: ', len(unidentified), flush=True)

    candidates = []
    for key in buckets.keys():
        bucket = buckets[key]
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                if bucket[i] < bucket[j]:
                    candidates.append((bucket[i], bucket[j]))
                elif bucket[i] > bucket[j]:
                    candidates.append((bucket[j], bucket[i]))
            if len(candidates) > 2000000:
                return candidates
    return candidates
