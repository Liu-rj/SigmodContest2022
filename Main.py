import pandas as pd
from FeatureExtracting import extract_x1, extract_x2
from EntityBlocking import block_x1, block_x2
from BucketBlocking import bucket_blocking
import time
from typing import *


def cal_recall(data, gnd):
    predict_pd = pd.read_csv('output.csv')
    gnd['cnt'] = 0
    predict = predict_pd.values.tolist()
    for idx in range(len(predict)):
        if predict[idx][0] == 0:
            break
        index = gnd[(gnd['lid'] == predict[idx][0]) & (gnd['rid'] == predict[idx][1])].index.tolist()
        if len(index) > 0:
            gnd['cnt'][index[0]] += 1
        if len(index) > 1:
            print('error')
            exit()
    print(sum(gnd['cnt']))
    print(sum(gnd['cnt']) / gnd.values.shape[0])
    left = gnd[gnd['cnt'] == 0].reset_index()
    # print that are uncovered
    # for idx in range(left.shape[0]):
    #     print(left['lid'][idx], ',', left['rid'][idx])
    #     print(data[data['id'] == left['lid'][idx]]['title'].iloc[0])
    #     print(data[data['id'] == left['rid'][idx]]['title'].iloc[0])
    #     print()


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
    output_df = pd.DataFrame(output, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)


brands_x1 = ['dell', 'lenovo', 'acer', 'asus', 'hp', 'panasonic', 'toshiba', 'sony', 'epson']
brands_x2 = ['sandisk', 'lexar', 'kingston', 'intenso', 'toshiba', 'sony', 'pny', 'samsung', '']
families_x2 = {'sandisk': ['extreme', 'cruzer', 'ultra', 'traveler', 'sdhc', 'usb', 'adapt'],
               'lexar': ['ultra', 'jumpdrive'],
               'toshiba': ['exceria', 'traveler', 'sdhc'],
               'kingston': ['traveler'],
               'sony': ['USM32GQX'],
               'intenso': ['premium', 'ultra', 'micro'],
               'pny': [],
               'samsung': [],
               '': ['microsdxc']}


def extract(data: pd.DataFrame):
    buckets: Dict[str, List] = {}
    for idx in range(data.shape[0]):
        title = data['title'][idx]
        key = '.'
        for brand in brands_x2:
            if brand in title:
                key = brand + key
                for name in families_x2[brand]:
                    if name in title:
                        key = key + name
                        break
                break
        if not key.startswith('.'):
            if key in buckets.keys():
                buckets[key].append((data['id'][idx], title))
            else:
                buckets[key] = [(data['id'][idx], title)]
    candidates = []
    for key in buckets.keys():
        bucket = buckets[key]
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                if bucket[i][0] < bucket[j][0]:
                    candidates.append((bucket[i][0], bucket[j][0]))
                elif bucket[i][0] > bucket[j][0]:
                    candidates.append((bucket[j][0], bucket[i][0]))
    output_df = pd.DataFrame(candidates, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)
    # cal_recall(raw_data, gnd_truth)


if __name__ == '__main__':
    raw_data = pd.read_csv('X2.csv')
    gnd_truth = pd.read_csv('Y2.csv')
    raw_data = raw_data[['id', 'name']]
    raw_data['name'] = raw_data.name.str.lower()
    # buckets = bucket_blocking(raw_data)
    # candidates = gen_candidates(buckets)
    # output_df = pd.DataFrame(candidates, columns=["left_instance_id", "right_instance_id"])
    # output_df.to_csv("output.csv", index=False)
    # cal_recall(raw_data, gnd_truth)

    features = extract_x2(raw_data)
    print(features[features['type'] != '0']['type'])

    # dataset = extract_x1(raw_data)
    # candidates_x1 = block_x1(dataset)
    #
    # candidates_x2 = [(0, 0)] * 2000000
    # save_output(candidates_x1, 1000000, candidates_x2, 2000000)
    # cal_recall(raw_data, gnd_truth)
