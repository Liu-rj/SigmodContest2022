from collections import defaultdict
from tqdm import tqdm
import re
from clean_x2 import clean_x2

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

lexar_patterns = [r'lexar|jumpdriver|s70|16gb',
                  r'lexar|c20m|jumpdrive|32gb']

kingston_patterns = [r'kingston|dt|101|g2|32gb']

intenso_patterns = [r'intenso|basic|line|16gb']

other_patterns = [r'pendrive|direct|sddd3-128g-g46|miniprice|sdxc',
                  r'memoria de|64gb 8gb|1000x|memory']
# patterns = ['sandisk|sdsqqnr|128g|zn6ia', 'sandisk|extreme|uhs-i|microsdxc|400gb', 'sandisk|automative|sdxc|64gb', 'sandisk|extreme|pro|sdxc|u3|512gb', 'sandisk|extreme|pro|microsdhc|uhs-i|16gb', 'sandisk|extreme|pro|microsdhc|uhs-l|32gb', 'sandisk|extreme|pro|microsdxc|uhs-i|64gb', 'sandisk|extreme|pro|microsdxc|uhs-il|64gb', 'sandisk|extreme|sdxc|uhs-i|class10|300x|128gb', 'sandisk|industrial|sdxc|64gb', 'sandisk|industrial|xl|sdxc|64gb', 'sandisk|industrial|microsdxc|128gb', 'sandisk|mobile|ultra|micro|sdhc|uhs-i|class10|16gb', 'sandisk|ultra|plus|microsdxc|uhs-i|a1|400gb', 'sandisk|extreme|compactflash|sdcfxs|128g', 'sandisk|extreme|compactflash|sdcfxs|32g', 'sandisk|extreme|compactflash|sdcfxs|64g', 'sandisk|extreme|compactflash|128gb|sdcfxs|128g', 'sandisk|extreme|compactflash|sdcfxs|16gb', 'sandisk|extreme|sdhc|sdxc|uhs-l|sd|8gb', 'sandisk|extreme|sdhc|16g|class10|60mb/s|16gb', 'sandisk|extreme|sdhc|64g|class10|60mb/s|64gb', 'sandisk|extreme|sdhc|16g|class10|90mb/s|16gb', 'sandisk|extreme|sdhc|32g|class10|90mb/s|32gb', 'sandisk|extreme|sdhc|class10|60mb/s|32gb', 'sandisk|extreme|sdxc|256g-class10-90mb/s|256gb', 'sandisk|extreme|sdxc|64g-class10-90mb/s|64gb', 'sandisk|extreme|sdxc|sd|class10|128gb', 'sandisk|extreme|microsdhc|uhs-l|16gb', 'sandisk|extreme|microsdxc|uhs-i|u3|128gb', 'sandisk|extreme|microsdxc|uhs-i|u3|64gb', 'sandisk|extreme|microsdhc|uhs-i|u3|32gb', 'sandisk|extreme|microsdhc|uhs-l|64gb', 'sandisk|extreme|pro|cfast2.0|128gb', 'sandisk|extreme|pro|cfast2.0|64gb', 'sandisk|extreme|pro|compactflash|16gb', 'sandisk|extreme|pro|compactflash|256gb', 'sandisk|extreme|pro|compactflash|32gb', 'sandisk|extreme|pro|compactflash|64gb', 'sandisk|extreme|pro|compactflash|128gb', 'sandisk|extreme|pro|compactflash|256gb', 'sandisk|extreme|pro|sdhc|uhs-i|u1|8gb', 'sandisk|extreme|pro|sdhc|uhs-i|u3|32gb', 'sandisk|extreme|pro|sdhc|uhs-ii|u3|32gb', 'sandisk|extreme|pro|sdhc|uhs-i|u3|16gb', 'sandisk|extreme|pro|sdxc|uhs-i|u1|64gb', 'sandisk|extreme|pro|sdxc|uhs-i|u3|256gb', 'sandisk|extreme|pro|sdxc|uhs-i|u3|512gb', 'sandisk|extreme|pro|sdxc|uhs-i|u3|64gb', 'sandisk|extreme|pro|sdxc|uhs-ii|u3|64gb', 'sandisk|extreme|pro|microsdhc|uhs-i|16gb', 'sandisk|ultra|sdhc|uhs-l|sd|16gb', 'sandisk|ultra|sdhc|uhs-l|sd|32gb', 'sandisk|ultra|sdhc|uhs-l|sd|64gb', 'sandisk|ultra|sdhc|sd|32g|class10|40mb/s', 'sandisk|ultra|sdhc|sd|8g|class10|40mb/s', 'sandisk|ultra|sdhc|sd|16g|class10', 'sandisk|ultra|plus|microsd|uhs-i|32gb', 'sandisk|ultra|plus|microsd|uhs-i|64gb', 'sandisk|ultra|microsd|uhs-i|class10|128gb', 'sandisk|ultra|microsd|uhs-i|class10|16gb', 'sandisk|ultra|microsd|uhs-i|class10|32gb', 'sandisk|ultra|microsdhc|uhs-i|a1|16gb', 'sandisk|ultra|microsdxc|uhs-i|a1|128gb', 'sandisk|ultra|microsdxc|uhs-i|a1|64gb', 'sandisk|ultra|microsdxc|uhs-i|128gb', 'sandisk|ultra|micro|sdhc|adapter|16gb', 'sandisk|ultra|micro|sdhc|adapter|32gb', 'sandisk|ultra|micro|sdhc|uhs-l|sd|64gb', 'sandisk|ultra|microsdhc|uhs-l|sd|32gb', 'sandisk|ultra|microsdhc|sd|class6|8gb', 'sandisk|ultra|microsdhc|sd|class6|16gb', 'sandisk|ultra|microsdhc|sd|class6|32gb', 'sandisk|ultra|microsdhc|sd|adapter|8gb', 'sandisk|ultra|microsdxc|uhs-l|class10|200gb', 'sandisk|ultra|microsdxc|uhs-l|class10|256gb', 'sandisk|extreme|microsdhc|uhs-i|sd|32gb', 'sandisk|extreme|pro|compactflash|128gb', 'sandisk|extreme|pro|sdxc|uhs-i|u3|128gb', 'sandisk|ultra|plus|microsd|uhs-i|128gb', 'sandisk|ultra|microsdhc|uhs-l|sd|16gb', 'sandisk|ultra|microsdhc|adapter|64gb']


def handler(X):
    X2 = clean_x2(X)
    name_full_map = defaultdict(list)
    confidence_map = defaultdict(list)
    pattern1_map = defaultdict(list)
    for i in tqdm(range(X2.shape[0])):
        name = X2['name'][i]
        capacity = X2['capacity'][i]
        brand = X2['brand'][i]

        # fully mapping pattern
        name_copy = name
        pattern_1 = sorted(set(name_copy.split()))
        name_full_map['_'.join(pattern_1)].append(i)

        # extremely special pattern
        flag = False
        if brand == 'sandisk':
            for pattern in sandisk_patterns:
                name_copy = name + ' ' + capacity
                result_re = sorted(set(re.findall(pattern, name_copy)))
                if len(result_re) == len(pattern.split('|')):
                    flag = True
                    confidence_map['_'.join(result_re)].append(i)
                    break

        elif brand == 'lexar':
            for pattern in lexar_patterns:
                name_copy = name + ' ' + capacity
                result_re = sorted(set(re.findall(pattern, name_copy)))
                if len(result_re) == len(pattern.split('|')):
                    flag = True
                    confidence_map['_'.join(result_re)].append(i)
                    break

        elif brand == 'kingston':
            for pattern in kingston_patterns:
                name_copy = name + ' ' + capacity
                if 'data' in name_copy and 'traveler' in name_copy:
                    name_copy += ' dt'
                if 'generation 2' in name_copy or 'gen2' in name_copy or 'gen 2' in name_copy or 'generación 2' in name_copy:
                    name_copy += ' g2'
                result_re = sorted(set(re.findall(pattern, name_copy)))
                if len(result_re) == len(pattern.split('|')):
                    flag = True
                    confidence_map['_'.join(result_re)].append(i)
                    break
        elif brand == 'intenso':
            for pattern in intenso_patterns:
                name_copy = name + ' ' + capacity
                if 'basic' in name_copy:
                    name_copy += ' basic line'
                result_re = sorted(set(re.findall(pattern, name_copy)))
                if len(result_re) == len(pattern.split('|')):
                    flag = True
                    confidence_map['_'.join(result_re)].append(i)
                    break

        # last method pattern
        if not flag:
            pattern1 = re.findall(r'[0-9]{2,}', name)
            pattern4 = re.findall(r'(?=[^\W\d_]*\d)(?=\d*[^\W\d_])[^\W_]{5,}', name)
            pattern = pattern4 + pattern1
            if len(pattern) == 0:
                continue
            pattern = list(sorted(pattern))
            pattern = [str(it).lower() for it in pattern]
            if X2['identification'][i] and brand == 'sandisk':
                key = X2['identification'][i] + '_'.join(pattern)
            elif X2['identification'][i]:
                key = X2['identification'][i]
            else:
                key = X2['brand'][i] + '_' + X2['instance_type'][i] + '_' + X2['capacity'][i] + '_' + '_'.join(pattern)
            pattern1_map[key].append(i)

    # add id pairs that share the same pattern to candidate set
    fully_mapping_pairs = []
    # for pattern in tqdm(name_full_map):
    #     ids = list(sorted(name_full_map[pattern]))
    #     if 1 < len(ids):
    #         for i in range(len(ids)):
    #             for j in range(i + 1, len(ids)):
    #                 fully_mapping_pairs.append((ids[i], ids[j]))

    # add id pairs that share the same pattern to candidate set
    candidate_pairs_1 = []
    # for pattern in tqdm(pattern1_map):
    #     ids = list(sorted(pattern1_map[pattern]))
    #     if 1 < len(ids) < 300:  # skip patterns that are too common
    #         for i in range(len(ids)):
    #             for j in range(i + 1, len(ids)):
    #                 candidate_pairs_1.append((ids[i], ids[j]))

    # add id pairs that share the same pattern to candidate set
    confidence_pairs = []
    for pattern in tqdm(confidence_map):
        ids = list(sorted(confidence_map[pattern]))
        if 1 < len(ids) < 300:  # skip patterns that are too common
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    confidence_pairs.append((ids[i], ids[j]))

    # remove duplicate pairs and take union
    confidence_pairs = set(confidence_pairs).union(set(fully_mapping_pairs))
    candidate_pairs = set(candidate_pairs_1)

    # 利用相似度筛去一些pair
    jaccard_similarities = []
    confidence_pairs_real_ids = []
    for id1, id2 in tqdm(confidence_pairs):
        real_id1 = X2['instance_id'][id1]
        real_id2 = X2['instance_id'][id2]
        # 因为此轮循环为confidence_pair，所以直接将相似度设置为1，代表完全置信
        if real_id1 < real_id2:
            confidence_pairs_real_ids.append((real_id1, real_id2))
            jaccard_similarities.append(1.0)
        else:
            confidence_pairs_real_ids.append((real_id2, real_id1))
            jaccard_similarities.append(1.0)

    candidate_pairs_real_ids = []
    for id1, id2 in tqdm(candidate_pairs):
        real_id1 = X2['instance_id'][id1]
        real_id2 = X2['instance_id'][id2]
        s1 = set(X2['name'][id1].split())
        s2 = set(X2['name'][id2].split())
        similarity = len(s1.intersection(s2)) / max(len(s1), len(s2))
        if similarity >= 0.4 and (real_id1, real_id2) not in confidence_pairs_real_ids:
            if real_id1 < real_id2:
                candidate_pairs_real_ids.append((real_id1, real_id2))
                jaccard_similarities.append(similarity)
            else:
                candidate_pairs_real_ids.append((real_id2, real_id1))
                jaccard_similarities.append(similarity)

    candidate_list = confidence_pairs_real_ids + candidate_pairs_real_ids
    candidate_list = [x for _, x in sorted(zip(jaccard_similarities, candidate_list), reverse=True)]
    print(len(candidate_list))
    return candidate_list
