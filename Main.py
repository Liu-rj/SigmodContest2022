import pandas as pd
from FeatureExtracting import extract_x2
from EntityBlocking import block_x2


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


brands_x1 = ['dell', 'lenovo', 'acer', 'asus', 'hp', 'panasonic', 'toshiba', 'sony', 'epson']


if __name__ == '__main__':
    raw_data = pd.read_csv('X1.csv')
    raw_data = pd.read_csv('X2.csv')
    raw_data['name'] = raw_data.name.str.lower()
    features = extract_x2(raw_data)
    candidates_x2 = block_x2(features)
    candidates_x1 = []
    save_output(candidates_x1, 1000000, candidates_x2, 2000000)
    # output_df = pd.DataFrame(candidates_x2, columns=['left_instance_id', 'right_instance_id'])
    # output_df.to_csv('output.csv', index=False)

    # gnd = pd.read_csv('Y2.csv')
    # gnd['cnt'] = 0
    # for idx in range(len(candidates_x2)):
    #     if candidates_x2[idx][0] == 0:
    #         continue
    #     index = gnd[(gnd['lid'] == candidates_x2[idx][0]) & (gnd['rid'] == candidates_x2[idx][1])].index.tolist()
    #     if len(index) > 0:
    #         if len(index) > 1:
    #             raise Exception
    #         gnd['cnt'][index[0]] += 1
    #     else:
    #         # print(candidates_x2[idx][2], '|', candidates_x2[idx][0][1], '|', candidates_x2[idx][1][1])
    #         pass
    # print('correct pairs:\t', sum(gnd['cnt']))
    # print('recall:\t\t\t', sum(gnd['cnt']) / gnd.values.shape[0])
    # left = gnd[gnd['cnt'] == 0].reset_index()
    # for idx in range(left.shape[0]):
    #     print(left['lid'][idx], ',', left['rid'][idx])
    #     print(raw_data[raw_data['id'] == left['lid'][idx]]['name'].iloc[0])
    #     print(raw_data[raw_data['id'] == left['rid'][idx]]['name'].iloc[0])
    #     print()
