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


if __name__ == '__main__':
    raw_data = pd.read_csv('X1.csv')
    raw_data = pd.read_csv('X2.csv')
    raw_data['name'] = raw_data.name.str.lower()
    features = extract_x2(raw_data)
    candidates_x2 = block_x2(features, 2000000)
    candidates_x1 = []
    save_output(candidates_x1, 1000000, candidates_x2, 2000000)
    # output_df = pd.DataFrame(candidates_x2, columns=['left_instance_id', 'right_instance_id'])
    # output_df.to_csv('output.csv', index=False)
