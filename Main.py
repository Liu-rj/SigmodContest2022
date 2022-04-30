import pandas as pd
from FeatureExtracting import extract_x2
from EntityBlocking import block_x2
from x1_nn_regex import *


def gen_candidates(buckets) -> []:
    candidates_pair = []
    for bucket in buckets:
        if len(bucket[1]) <= 1:
            continue
        for idx_1 in range(len(bucket[1])):
            for idx_2 in range(idx_1 + 1, len(bucket[1])):
                if bucket[1][idx_1] < bucket[1][idx_2]:
                    candidates_pair.append((bucket[1][idx_1], bucket[1][idx_2]))
                else:
                    candidates_pair.append((bucket[1][idx_2], bucket[1][idx_1]))
    return candidates_pair


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
    mode = 0
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'True'
    if mode == 0:
        path = './fromstart_further_x1_berttiny_finetune_epoch20_margin0.01'
        raw_data = pd.read_csv("../million_x1.csv")
        raw_data['title'] = raw_data.title.str.lower()
        candidates_x1 = x1_test(raw_data, 1000000, path)
        raw_data = pd.read_csv('X2.csv')
        raw_data['name'] = raw_data.name.str.lower()
        features = extract_x2(raw_data)
        candidates_x2 = block_x2(features, 2000000)
        print('success')
        save_output(candidates_x1, 1000000, candidates_x2, 2000000)
    elif mode == 1:
        import time
        start = time.time()
        path = './fromstart_further_x1_berttiny_finetune_epoch20_margin0.01'
        raw_data = pd.read_csv("X1.csv")
        raw_data['title'] = raw_data.title.str.lower()
        candidates_x1 = x1_test(raw_data, 2814, path)
        raw_data = pd.read_csv('X2.csv')
        raw_data['name'] = raw_data.name.str.lower()
        features = extract_x2(raw_data)
        candidates_x2 = block_x2(features, 4392)
        print('success')
        end = time.time()
        print('time:', end - start)
        candidate_pairs = candidates_x2
        raw_data = pd.read_csv('X2.csv')
        gnd = pd.read_csv('Y2.csv')
        gnd['cnt'] = 0
        brands = ['sandisk', 'lexar', 'kingston', 'intenso', 'toshiba', 'sony', 'pny', 'samsung', 'transcend', '']
        print('-----------------------------------------------------------------------------------------------')
        cnt_dict: Dict[str, int] = defaultdict(int)
        for idx in range(len(candidate_pairs)):
            left_id = candidate_pairs[idx][0]
            right_id = candidate_pairs[idx][1]
            index = gnd[(gnd['lid'] == left_id) & (gnd['rid'] == right_id)].index.tolist()
            if len(index) > 0:
                if len(index) > 1:
                    raise Exception
                gnd['cnt'][index[0]] += 1
                if gnd['cnt'][index[0]] > 1:
                    print(index)
            else:
                # wrong pairs
                left_sentence = raw_data[raw_data['id'] == left_id]['name'].values[0]
                right_sentence = raw_data[raw_data['id'] == right_id]['name'].values[0]
                for b in brands:
                    if b in left_sentence.lower() or b in right_sentence.lower():
                        cnt_dict[b] += 1
                        if b == 'samsung':
                            # print(left_id, '|', right_id)
                            # print(left_sentence, '|', right_sentence)
                            pass
                        break
                # if left_text != right_text:
                #     print(idx, left_id, right_id)
                #     print(left_text, '|', right_text)
                pass
        for key in cnt_dict.keys():
            cnt_dict[key] = cnt_dict[key] / gnd.shape[0]
        print('wrong pairs:', cnt_dict)
        print('-----------------------------------------------------------------------------------------------')
        left = gnd[gnd['cnt'] == 0]
        cnt_dict: Dict[str, int] = defaultdict(int)
        for idx in left.index:
            # unrecognized pairs
            left_sentence = raw_data[raw_data['id'] == left['lid'][idx]]['name'].iloc[0]
            right_sentence = raw_data[raw_data['id'] == left['rid'][idx]]['name'].iloc[0]
            for b in brands:
                if b in left_sentence.lower() or b in right_sentence.lower():
                    cnt_dict[b] += 1
                    if b == 'kingston':
                        # print(left_sentence, '|', right_sentence)
                        pass
                    break
        for key in cnt_dict.keys():
            cnt_dict[key] = cnt_dict[key] / gnd.shape[0]
        print('not recognized pairs:', cnt_dict)
        print('output pairs:\t', len(candidate_pairs))
        print('correct pairs:\t', sum(gnd['cnt']))
        print('recall:\t\t\t', sum(gnd['cnt']) / gnd.values.shape[0])
        
        origin_data = pd.read_csv("./X1.csv")
        origin_gnd = pd.read_csv("./Y1.csv")
        origin_data['title'] = origin_data.title.str.lower()
        origin_data = origin_data[['id', 'title']]
        origin_pairs = x1_test(origin_data, 2814, path)
        print("Model: %s, origin recall: %f" % (path, recall_calculation(origin_pairs, origin_gnd)))
