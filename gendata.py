from sentence_transformers import SentenceTransformer, util
import pandas as pd
from typing import *
import random


# raw_data = pd.read_csv('data/Sigmod/X1.csv')
# for i in range(10):
#     raw_data = pd.concat([raw_data, raw_data], axis=0, ignore_index=True)
# raw_data.to_csv('data/X1_BIG.csv', index=False)

# raw_data = pd.read_csv('data/X1_BIG.csv')
# raw_data = raw_data[:1000000]
# raw_data.to_csv('data/X1_BIG.csv', index=False)


def back_union(Y) -> List[Set]:
    unions: List[Set[int]] = []
    for i in range(Y.shape[0]):
        found = False
        candidates = []
        for union in unions:
            if Y['lid'][i] in union or Y['rid'][i] in union:
                union.add(Y['lid'][i])
                union.add(Y['rid'][i])
                found = True
                candidates.append(union)
        if len(candidates) > 1:
            union = set()
            for candidate in candidates:
                unions.remove(candidate)
                union = union.union(candidate)
            unions.append(union)
        if not found:
            unions.append({Y['lid'][i], Y['rid'][i]})
    return unions


def cal_recall(gnd):
    predict_pd = pd.read_csv('output.csv')
    gnd['cnt'] = 0
    predict = predict_pd.values.tolist()
    for idx in range(len(predict)):
        if predict[idx][0] == 0:
            break
        index = gnd[(gnd['lid'] == predict[idx][0]) & (
                gnd['rid'] == predict[idx][1])].index.tolist()
        if len(index) > 0:
            gnd['cnt'][index[0]] += 1
        if len(index) > 1:
            print('error')
            exit()
    print(sum(gnd['cnt']))
    print(sum(gnd['cnt']) / gnd.values.shape[0])


model = SentenceTransformer('./model/mix_base')

# Single list of sentences - Possible tens of thousands of sentences
# sentences = ['The cat sits outside',
#              'A man is playing guitar',
#              'I love pasta',
#              'The new movie is awesome',
#              'The cat plays in the garden',
#              'A woman watches TV',
#              'In the garden, a cat sits',
#              'The new movie is so great',
#              'Do you like pizza?']

raw_data = pd.read_csv('X2.csv')
ground = pd.read_csv('Y2.csv')
raw_data['title'] = raw_data['name']
unions_id = back_union(ground)
union_titles: List[List[str]] = []
for union in unions_id:
    bucket = []
    for ele in union:
        bucket.append(raw_data[raw_data['id'] == ele]['title'].iloc[0])
    union_titles.append(bucket)


raw_data['title'] = raw_data.title.str.lower()

sentences = raw_data['title'].drop_duplicates(
    keep='first').reset_index(drop=True)

print(len(sentences))
paraphrases = util.paraphrase_mining(
    model, sentences, show_progress_bar=True, batch_size=64)

print(len(paraphrases))
print(paraphrases[-1])

# candidates = set()
mismatches: List[Tuple[str, str, int]] = []
for paraphrase in paraphrases:
    score, i, j = paraphrase
    match = False
    for union in union_titles:
        if sentences[i] in union and sentences[j] in union:
            match = True
            break
    if not match:
        mismatches.append((sentences[i], sentences[j], 0))
    if score <= 0.5:
        break
    # lids = raw_data[raw_data['title'] == sentences[i]]['id']
    # rids = raw_data[raw_data['title'] == sentences[j]]['id']
    # for lid in lids:
    #     for rid in rids:
    #         if lid < rid:
    #             candidates.add((lid, rid))
    #         elif lid > rid:
    #             candidates.add((rid, lid))
    # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))

matches: List[Tuple[str, str, int]] = []
# mismatches: List[Tuple[str, str, int]] = []
for k in range(len(union_titles)):
    union = union_titles[k]
    for i in range(len(union)):
        for j in range(i + 1, len(union)):
            matches.append((union[i], union[j], 1))
    # for idx in range(k + 1, len(union_titles)):
    #     for title_1 in union:
    #         for title_2 in union_titles[idx]:
    #             mismatches.append((title_1, title_2, 0))
print(len(matches))
print(len(mismatches))
random.shuffle(matches)
random.shuffle(mismatches)
debris_ma = int(len(matches) / 7)
debris_mi = int(len(mismatches) / 7)
print(debris_ma, debris_mi)
train_set = matches[:5 * debris_ma] + mismatches[:5 * debris_mi]
dev_set = matches[5 * debris_ma:6 * debris_ma] + \
          mismatches[5 * debris_mi:6 * debris_mi]
test_set = matches[6 * debris_ma:] + mismatches[6 * debris_mi:]
random.shuffle(train_set)
random.shuffle(dev_set)
random.shuffle(test_set)
output_df = pd.DataFrame(train_set, columns=['left', 'right', 'score'])
output_df.to_csv('data/sts_train_x2.csv', index=False)
output_df = pd.DataFrame(dev_set, columns=['left', 'right', 'score'])
output_df.to_csv('data/sts_dev_x2.csv', index=False)
output_df = pd.DataFrame(test_set, columns=['left', 'right', 'score'])
output_df.to_csv('data/sts_test_x2.csv', index=False)

# output_df = pd.DataFrame(candidates, columns=["left_instance_id", "right_instance_id"])
# output_df.to_csv("output.csv", index=False)
# Y = pd.read_csv('data/Sigmod/Y1.csv')
# cal_recall(Y)
#
# # para_df = pd.DataFrame(paraphrases, columns=['similarity', 'left', 'right'])
# # print(para_df)
