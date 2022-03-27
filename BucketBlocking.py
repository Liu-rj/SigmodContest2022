import pandas as pd


def cal_recall():
    predict_pd = pd.read_csv('output.csv')
    gnd = pd.read_csv('Y1.csv')
    predict = predict_pd.values.tolist()
    cnt = 0
    for idx in range(len(predict)):
        if not gnd[(gnd['lid'] == predict[idx][0]) & (gnd['rid'] == predict[idx][1])].empty:
            cnt += 1
    print(cnt)
    print(cnt / gnd.values.shape[0])


def output(pairs, size):
    if len(pairs) > size:
        pairs = pairs[:size]
    output_df = pd.DataFrame(pairs, columns=["left_instance_id", "right_instance_id"])
    output_df.to_csv("output.csv", index=False)
    cal_recall()


if __name__ == '__main__':
    nonsense = ['|', ',', '-', ':', '/', '+', '&']
    X = pd.read_csv('X1.csv')
    Y = pd.read_csv('Y1.csv')
    buckets: list[tuple[set[str], list[int]]] = []
    for i in range(X.shape[0]):
        found = False
        vector = set([it for it in X['title'][i].lower().split() if it not in nonsense])
        for bucket in buckets:
            jaccard_similarity = len(vector.intersection(bucket[0])) / max(len(vector), len(bucket[0]))
            if jaccard_similarity >= 0.8:
                bucket[1].append(X['id'][i])
                found = True
                break
        if not found:
            buckets.append((vector, [X['id'][i]]))
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
    expected_cand_size = 1000000
    output(candidates_pair, expected_cand_size)
