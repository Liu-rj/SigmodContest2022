import pandas as pd
from BackUnion import back_union
from FeatureExtracting import extract_x1, extract_x2
from EntityBlocking import block_x1, block_x2
import time


def cal_recall(data, gnd):
    predict_pd = pd.read_csv('output.csv')
    # unions = back_union(gnd)
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
    # left = gnd[gnd['cnt'] == 0].reset_index()
    # with open('records/weird.txt', 'w', encoding='utf-8') as file:
    #     for idx in range(left.shape[0]):
    #         print(left['lid'][idx], ',', left['rid'][idx])
    #         print(data[data['id'] == left['lid'][idx]].iloc[0])
    #         print(data[data['id'] == left['rid'][idx]].iloc[0])
    #         print()
    #         for union in unions:
    #             if left['lid'][idx] in union or left['rid'][idx] in union:
    #                 file.write(str(union) + '\n')
    #                 for it in union:
    #                     file.write(data[data['id'] == it]['title'].iloc[0] + '\n')
    #                 file.write('\n')
    #                 unions.remove(union)
    #                 break


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


brands = ['dell', 'lenovo', 'acer', 'asus', 'hp', 'panasonic', 'toshiba', 'sony', 'epson']


def extract(data: pd.DataFrame):
    time1 = time.time()
    titles = data['title']
    ids = data['id']
    num = 0
    for i in range(len(titles)):
        have_brand = False
        for brand in brands:
            if brand in titles[i].lower():
                have_brand = True
                break
        if not have_brand:
            num += 1
            print(ids[i], titles[i])
    time2 = time.time()
    print(num)
    print(time2 - time1)
    # data['brand'] = data.apply(lambda x: brands[0] in x.title.lower(), axis=1)
    # for brand in brands:
    #     data[brand] = data.apply(lambda x: brand in x.title.lower(), axis=1)
    # print(data)


if __name__ == '__main__':
    raw_data = pd.read_csv('X1.csv')
    gnd_truth = pd.read_csv('Y1.csv')
    extract(raw_data)
    # dataset = extract_x1(raw_data)
    # candidates_x1 = block_x1(dataset)
    #
    # candidates_x2 = [(0, 0)] * 2000000
    # save_output(candidates_x1, 1000000, candidates_x2, 2000000)
    # cal_recall(raw_data, gnd_truth)
