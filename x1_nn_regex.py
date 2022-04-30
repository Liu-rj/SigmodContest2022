import os, multiprocessing as mp
import textdistance as td
from collections import defaultdict


def exact_eq(x, y):
    return float(x == y)


def token_ops(func):
    def new_func(x, y):
        return func(x.split(), y.split())

    return new_func


def set_jaccard(x, y):
    if type(x) is dict:
        x1 = set(x.keys())
        y1 = set(y.keys())
        return len(x1.intersection(y1)) / max(len(x1), len(y1))
    else:
        return len(x.intersection(y)) / max(len(x), len(y))


def str_eq(x: str, y: str):
    return x == y


SIM_FUNC_DICT = {
    ("title", "jaccard"): token_ops(td.jaccard.normalized_similarity),
    ("title", "overlap"): token_ops(td.overlap.normalized_similarity),
    # ("title", "damerau_levenshtein"): td.damerau_levenshtein.normalized_similarity,
    ("title", "jaro_winkler"): td.jaro_winkler.normalized_similarity,
    # ("brand", "set_jaccard"): set_jaccard,
    # ("cpu_model", "set_jaccard"): set_jaccard,
    # ("ram_capacity", "equals"): str_eq,
    # ("family", "equals"): str_eq,
    ("pc_name", "equals"): str_eq
}


def record_sim_func(record_pair):
    record_left, record_right = record_pair
    feature_dict = {}

    for (field, sim_func_name), sim_func in SIM_FUNC_DICT.items():
        x = record_left[field]
        y = record_right[field]
        if x and y:
            if sim_func_name.startswith('abs_diff') and (math.isnan(x) or math.isnan(y)):
                sim = -1.0
            else:
                sim = sim_func(x, y)
        else:
            sim = -1.0
        feature_dict[f"{field}_{sim_func_name}"] = sim

    return feature_dict


# def compare_pairs(record_dict, found_pair_set):
#     all_feature_dict = defaultdict(list)
#     chunksize = 50000
#     tasks = (
#         (record_dict.iloc[id_left], record_dict.iloc[id_right])
#         # for (id_left,id_right,_,_)
#         for (id_left, id_right, _)
#         in found_pair_set
#     )

    # with multiprocessing.Pool() as pool:
    #     for feature_dict in tqdm(
    #             pool.imap(record_sim_func, tasks, chunksize=chunksize),
    #             total=len(found_pair_set)
    #     ):
    #         print('new process created.')
    #         for feature, val in feature_dict.items():
    #             all_feature_dict[feature].append(val)
    #
    #     pool.close()
    #     pool.join()

    # found_pair_set=pd.DataFrame(found_pair_set,columns=['left_id','right_id','cosine','label'])
    # found_pair_set = pd.DataFrame(found_pair_set, columns=['left_id', 'right_id', 'cosine'])
    # # return pd.DataFrame(all_feature_dict, index=pd.MultiIndex.from_tuples(found_pair_set))\
    # all_feature_dict = pd.DataFrame(all_feature_dict)
    # return pd.concat([found_pair_set, all_feature_dict], axis=1)


def recall_calculation(predict: list, gnd):
    cnt = 0
    for i in range(len(predict)):
        if not gnd[(gnd['lid'] == predict[i][0]) & (gnd['rid'] == predict[i][1])].empty:
            cnt += 1
    return cnt / gnd.values.shape[0]


def dispatch_task(feature_df, found_pair_set, nproc, q: mp.Queue, res_q: mp.Queue):
    idx = 0
    # feature_df.set_index
    title_l = feature_df['title'].values
    pcname_l = feature_df['pc_name'].values
    step = int(len(found_pair_set) / nprocs)
    tasks = [
        ({'title': title_l[id_left], 'pc_name': pcname_l[id_left]}, {'title': title_l[id_right], 'pc_name': pcname_l[id_right]})
        for (id_left, id_right, _)
        in found_pair_set
    ]
    # print(step)
    # print(tasks)
    # print(feature_df)
    # print(found_pair_set)
    for rank in range(nprocs):
        if rank == nproc - 1:
            q.put(tasks[idx:])
        else:
            q.put(tasks[idx:idx + step])
            idx += step
    all_feature_dict = defaultdict(list)
    for rank in range(nproc):
        feature_dict = res_q.get()
        for feature, val in feature_dict.items():
                all_feature_dict[feature].extend(val)
    found_pair_set = pd.DataFrame(found_pair_set, columns=['left_id', 'right_id', 'cosine'])
    all_feature_dict = pd.DataFrame(all_feature_dict)
    return pd.concat([found_pair_set, all_feature_dict], axis=1)


def wait_child_proc(q: mp.Queue, res_queue: mp.Queue):
    tasks = q.get()
    print('child process get task')
    # print(len(tasks))
    all_feature_dict = defaultdict(list)
    for task in tasks:
        features = record_sim_func(task)
        for feature, val in features.items():
            all_feature_dict[feature].append(val)
    print('child process complete task')
    res_queue.put(all_feature_dict)


def x1_test(data, all_limit: int, model_path: str, nproc, q, res_q) -> list:
    print('enter x1_t[est function')
    # clusters = handle(data)
    features = clean(data)
    model = SentenceTransformer(model_path, device='cpu')
    encodings = model.encode(sentences=data['title'], batch_size=256, normalize_embeddings=True)
    topk = 50
    candidate_pairs: List[Tuple[int, int, float]] = []
    ram_capacity_list = features['ram_capacity'].values
    cpu_model_list = features['cpu_model'].values
    title_list = features['title'].values
    family_list = features['family'].values
    identification_list = defaultdict(list)
    reg_list = defaultdict(list)
    number_list = defaultdict(list)
    regex_pattern = re.compile('(?=[^\W\d_]*\d)(?=\d*[^\W\d_])[^\W_gGM]{6,}')
    number_pattern = re.compile('[0-9]{6,}')
    buckets = defaultdict(list)
    for idx in range(data.shape[0]):
        title = " ".join(sorted(set(title_list[idx].split())))

        regs = regex_pattern.findall(title)
        identification = " ".join(sorted(regs))
        reg_list[identification].append(idx)

        identification_list[title].append(idx)

        number_id = number_pattern.findall(title)
        number_id = " ".join(sorted(number_id))
        number_list[number_id].append(idx)

        brands = features['brand'][idx]
        for brand in brands:
            buckets[brand].append(idx)
        if len(brands) == 0:
            buckets['0'].append(idx)
    visited_set = set()
    ids = data['id'].values

    regex_pairs = []
    for key in identification_list:
        cluster = identification_list[key]
        if len(cluster) > 1:
            for i in range(0, len(cluster) - 1):
                for j in range(i + 1, len(cluster)):
                    s1 = ids[cluster[i]]
                    s2 = ids[cluster[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    token = str(small) + " " + str(large)
                    if token in visited_set:
                        continue
                    visited_set.add(token)
                    regex_pairs.append((small, large))
    for key in reg_list:
        cluster = reg_list[key]
        if len(cluster) <= 3:
            for i in range(0, len(cluster) - 1):
                for j in range(i + 1, len(cluster)):
                    s1 = ids[cluster[i]]
                    s2 = ids[cluster[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    token = str(small) + " " + str(large)
                    if token in visited_set:
                        continue
                    visited_set.add(token)
                    regex_pairs.append((small, large))
    for key in number_list:
        cluster = number_list[key]
        if len(cluster) <= 3:
            for i in range(0, len(cluster) - 1):
                for j in range(i + 1, len(cluster)):
                    s1 = ids[cluster[i]]
                    s2 = ids[cluster[j]]
                    small = min(s1, s2)
                    large = max(s1, s2)
                    token = str(small) + " " + str(large)
                    if token in visited_set:
                        continue
                    visited_set.add(token)
                    regex_pairs.append((small, large))
    limit = all_limit - len(regex_pairs)
    if limit < 0:
        limit = 0
    for key in buckets:
        cluster = buckets[key]
        embedding_matrix = encodings[cluster]
        k = min(topk, len(cluster))
        index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
        index_model.hnsw.efConstruction = 100
        index_model.add(embedding_matrix)
        index_model.hnsw.efSearch = 256
        D, I = index_model.search(embedding_matrix, k)
        for i in range(len(D)):
            for j in range(len(D[0])):
                index1 = cluster[i]
                index2 = cluster[I[i][j]]
                s1 = ids[index1]
                s2 = ids[index2]
                if s1 == s2:
                    continue
                small = min(s1, s2)
                large = max(s1, s2)
                visit_token = str(small) + " " + str(large)
                if visit_token in visited_set:
                    continue
                visited_set.add(visit_token)
                if not (ram_capacity_list[index1] == '0' or ram_capacity_list[index2] == '0' or ram_capacity_list[
                    index1] == ram_capacity_list[index2]):
                    if family_list[index1] != 'x220' and family_list[index2] != 'x220':
                        continue
                intersect = cpu_model_list[index1].intersection(cpu_model_list[index2])
                if not (len(cpu_model_list[index1]) == 0 or len(cpu_model_list[index2]) == 0 or len(intersect) != 0):
                    continue
                # candidate_pairs.append((small, large, D[i][j]))
                candidate_pairs.append((index1, index2, D[i][j]))

    candidate_pairs.sort(key=lambda x: x[2])
    # candidate_pairs = candidate_pairs[:limit]
    half_limit = int(0.5 * limit)
    rfc_pairs = [(x[0], x[1]) for x in candidate_pairs[:half_limit]]
    # rfc_pairs = list(map(lambda x: (x[0], x[1]), candidate_pairs[0:half_limit]))
    # output.extend(regex_pairs)
    output = regex_pairs
    if len(candidate_pairs)>half_limit:
        f = open("./RFC_measure_jaje_2.bin", 'rb')
        clf = pickle.load(f)
        f.close()
        remained_pairs = candidate_pairs[half_limit:half_limit + 1500000]
        train_pairs = dispatch_task(features, remained_pairs, nproc, q, res_q)
        # train_pairs = compare_pairs(features, remained_pairs)
        measures = train_pairs.drop(columns=['left_id', 'right_id'], inplace=False)
        y_pred = clf.predict_proba(measures)
        y_pred[y_pred >= 0.15] = 1
        y_pred[y_pred < 0.15] = 0
        y_pred = y_pred[:, 1]

        left = train_pairs['left_id'].values
        right = train_pairs['right_id'].values
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                rfc_pairs.append((left[i], right[i]))

    for pair in rfc_pairs:
        s1 = ids[pair[0]]
        s2 = ids[pair[1]]
        small = min(s1, s2)
        large = max(s1, s2)
        if small == large:
            continue
        output.append((small, large))
        if len(output) == all_limit:
            break
    return output


def save_output(X1_candidate_pairs,
                X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csvcpu_model=set(list(map(lambda x:x[1:],cpu_model_list)))
    expected_cand_size_X1 = 1000000
    expected_cand_size_X2 = 2000000

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) > expected_cand_size_X1:
        X1_candidate_pairs = X1_candidate_pairs[:expected_cand_size_X1]
    if len(X2_candidate_pairs) > expected_cand_size_X2:
        X2_candidate_pairs = X2_candidate_pairs[:expected_cand_size_X2]

    # make sure to include exactly 1000000 pairs for dataset X1 and 2000000 pairs for dataset X2
    if len(X1_candidate_pairs) < expected_cand_size_X1:
        X1_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X1 - len(X1_candidate_pairs)))
    if len(X2_candidate_pairs) < expected_cand_size_X2:
        X2_candidate_pairs.extend([(0, 0)] * (expected_cand_size_X2 - len(X2_candidate_pairs)))

    all_cand_pairs = X1_candidate_pairs + X2_candidate_pairs  # make sure to have the pairs in the first dataset first
    output_df = pd.DataFrame(all_cand_pairs, columns=["left_instance_id", "right_instance_id"])
    # In evaluation, we expect output.csv to include exactly 3000000 tuple pairs.
    # we expect the first 1000000 pairs are for dataset X1, and the remaining pairs are for dataset X2
    output_df.to_csv("output.csv", index=False)


if __name__ == '__main__':
    path = './fromstart_further_x1_berttiny_finetune_epoch20_margin0.01'
    os.environ['TOKENIZERS_PARALLELISM'] = 'True'
    mode = 0
    if mode == 0:
        nprocs = 28
        queue = mp.Queue()
        res_queue = mp.Queue()
        process_list = []
        for rank in range(nprocs):
            if rank == nprocs - 1:
                process = mp.Process(target=wait_child_proc, args=(queue, res_queue))
            else:
                process = mp.Process(target=wait_child_proc, args=(queue, res_queue))
            process_list.append(process)
            process.start()
        

        import re
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        import faiss
        from tqdm import tqdm
        from typing import *
        from clean import clean
        import pickle
        import math
        from tqdm.auto import tqdm
        raw_data = pd.read_csv("X1.csv")
        # raw_data = raw_data
        raw_data['title'] = raw_data.title.str.lower()
        # import time
        # print(time.time())
        x1_pairs = x1_test(raw_data, 1000000, path, nprocs, queue, res_queue)
        # print(time.time())
        raw_data = pd.read_csv("X2.csv")
        save_output(x1_pairs, [])
        print("success")

        for process in process_list:
            process.join()
        # calculate
        # with open('Y1.csv', 'r') as csv1, open('output.csv', 'r') as csv2:
        #     import1 = csv1.readlines()
        #     import2 = csv2.readlines()
        #     same = 0
        #     for row in import2:
        #         if row in import1:
        #             same = same + 1
        #     print(same / len(import1))
    # elif mode == 1:
    #     test_data = pd.read_csv("./data/x1_test.csv")
    #     train_data = pd.read_csv("./data/x1_train.csv")
    #     origin_data = pd.read_csv("./X1.csv")
    #     test_gnd = pd.read_csv("./data/y1_test.csv")
    #     train_gnd = pd.read_csv("./data/y1_train.csv")
    #     origin_gnd = pd.read_csv("./Y1.csv")
    #     test_data['title'] = test_data.title.str.lower()
    #     train_data['title'] = train_data.title.str.lower()
    #     origin_data['title'] = origin_data.title.str.lower()
    #     # test_data['instance_id']=test_data['id']
    #     # train_data['instance_id']=train_data['id']
    #     # origin_data['instance_id']=origin_data['id']
    #     test_data = test_data[['id', 'title']]
    #     train_data = train_data[['id', 'title']]
    #     origin_data = origin_data[['id', 'title']]
    #     test_pairs = x1_test(test_data, 488, path)
    #     train_pairs = x1_test(train_data, 2326, path)
    #     origin_pairs = x1_test(origin_data, 2814, path)
    #     # raw_data = pd.read_csv('../X1.csv')
    #     # gnd = pd.read_csv('../Y1.csv')
    #     # gnd['cnt'] = 0
    #     # features = clean(raw_data)
    #     # for idx in range(len(origin_pairs)):
    #     #     left_id = origin_pairs[idx][0]
    #     #     right_id = origin_pairs[idx][1]
    #     #     index = gnd[(gnd['lid'] == left_id) & (gnd['rid'] == right_id)].index.tolist()
    #     #     if len(index) > 0:
    #     #         if len(index) > 1:
    #     #             raise Exception
    #     #         gnd['cnt'][index[0]] += 1
    #     #         if gnd['cnt'][index[0]] > 1:
    #     #             print(index)
    #     #     else:
    #     #         left_text = raw_data[raw_data['id'] == left_id]['title'].values[0]
    #     #         right_text = raw_data[raw_data['id'] == right_id]['title'].values[0]
    #     #         if left_text != right_text:
    #     #             print(idx, left_id, right_id)
    #     #             print(left_text, '|', right_text)
    #     #             print(features[features['instance_id'] == left_id]['brand'].iloc[0], '|||',
    #     #                   features[features['instance_id'] == right_id]['brand'].iloc[0])
    #     #             print(features[features['instance_id'] == left_id]['family'].iloc[0], '|||',
    #     #                   features[features['instance_id'] == right_id]['family'].iloc[0])
    #     #             print(features[features['instance_id'] == left_id]['cpu_model'].iloc[0], '|||',
    #     #                   features[features['instance_id'] == right_id]['cpu_model'].iloc[0])
    #     #             print(features[features['instance_id'] == left_id]['pc_name'].iloc[0], '|||',
    #     #                   features[features['instance_id'] == right_id]['pc_name'].iloc[0])
    #     #         pass
    #     # print('-----------------------------------------------------------------------------------------------')
    #     # left = gnd[gnd['cnt'] == 0]
    #     # for idx in left.index:
    #     #     if features[features['instance_id'] == left['lid'][idx]]['brand'].iloc[0] != \
    #     #             features[features['instance_id'] == left['rid'][idx]]['brand'].iloc[0]:
    #     #         print(left['lid'][idx], ',', left['rid'][idx])
    #     #     title1 = raw_data[raw_data['id'] == left['lid'][idx]]['title'].iloc[0]
    #     #     title2 = raw_data[raw_data['id'] == left['rid'][idx]]['title'].iloc[0]
    #     #     title1 = " ".join(sorted(title1.split()))
    #     #     title2 = " ".join(sorted(title2.split()))
    #     #     print(title1)
    #     #     print(title2)
    #     #     print(raw_data[raw_data['id'] == left['lid'][idx]]['title'].iloc[0], '|||',
    #     #           raw_data[raw_data['id'] == left['rid'][idx]]['title'].iloc[0])
    #     print("Model: %s, test recall: %f, train recall: %f, origin recall: %f" % (
    #         path, recall_calculation(test_pairs, test_gnd), recall_calculation(train_pairs, train_gnd),
    #         recall_calculation(origin_pairs, origin_gnd)))
    # elif mode == 2:
    #     # import os
    #     # os.environ['TOKENIZERS_PARALLELISM'] = 'True'
    #     # data = pd.read_csv("./data/x1_train.csv")
    #     # features = clean(data)
    #     # model = SentenceTransformer(path, device='cpu')
    #     # encodings = model.encode(sentences=data['title'], batch_size=256, normalize_embeddings=True)
    #     # topk = 50
    #     # buckets = defaultdict(list)
    #     # for idx in range(data.shape[0]):
    #     #     brands = features['brand'][idx]
    #     #     for brand in brands:
    #     #         buckets[brand].append(idx)
    #     #     if len(brands) == 0:
    #     #         buckets['0'].append(idx)
    #     # ids = data['id'].values
    #     # visited_set = set()
    #     # ram_capacity_list = features['ram_capacity'].values
    #     # family_list = features['family'].values
    #     # cpu_model_list = features['cpu_model'].values
    #     # title_list = features['title'].values
    #     # brand_list = features['brand'].values
    #     # pcname_list = features['pc_name'].values
    #     # candidate_pairs = []
    #     # for key in buckets:
    #     #     cluster = buckets[key]
    #     #     embedding_matrix = encodings[cluster]
    #     #     k = min(topk, len(cluster))
    #     #     index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
    #     #     index_model.hnsw.efConstruction = 100
    #     #     index_model.add(embedding_matrix)
    #     #     index_model.hnsw.efSearch = 256
    #     #     D, I = index_model.search(embedding_matrix, k)
    #     #     for i in range(len(D)):
    #     #         for j in range(len(D[0])):
    #     #             index1 = cluster[i]
    #     #             index2 = cluster[I[i][j]]
    #     #             s1 = ids[index1]
    #     #             s2 = ids[index2]
    #     #             if s1 == s2:
    #     #                 continue
    #     #             small = min(s1, s2)
    #     #             large = max(s1, s2)
    #     #             visit_token = str(small) + " " + str(large)
    #     #             if visit_token in visited_set:
    #     #                 continue
    #     #             visited_set.add(visit_token)
    #     #             if not (ram_capacity_list[index1] == '0' or ram_capacity_list[index2] == '0' or ram_capacity_list[
    #     #                 index1] == ram_capacity_list[index2]):
    #     #                 if family_list[index1] != 'x220' and family_list[index2] != 'x220':
    #     #                     continue
    #     #             intersect = cpu_model_list[index1].intersection(cpu_model_list[index2])
    #     #             if not (len(cpu_model_list[index1]) == 0 or len(cpu_model_list[index2]) == 0 or len(
    #     #                     intersect) != 0):
    #     #                 continue
    #     #             candidate_pairs.append([index1, index2, D[i][j]])
    #     #
    #     # train_data = []
    #     # gnd = pd.read_csv("./data/y1_train.csv")
    #     # for i in range(len(candidate_pairs)):
    #     #     s1 = ids[candidate_pairs[i][0]]
    #     #     s2 = ids[candidate_pairs[i][1]]
    #     #     small = min(s1, s2)
    #     #     large = max(s1, s2)
    #     #     if not gnd[(gnd['lid'] == small) & (gnd['rid'] == large)].empty:
    #     #         train_data.append(candidate_pairs[i] + [1])
    #     #     else:
    #     #         train_data.append(candidate_pairs[i] + [0])
    #
    #     import textdistance as td
    #     import math
    #
    #
    #     def exact_eq(x, y):
    #         return float(x == y)
    #
    #
    #     def token_ops(func):
    #         def new_func(x, y):
    #             return func(x.split(), y.split())
    #
    #         return new_func
    #
    #
    #     def set_jaccard(x, y):
    #         if type(x) is dict:
    #             x1 = set(x.keys())
    #             y1 = set(y.keys())
    #             return len(x1.intersection(y1)) / max(len(x1), len(y1))
    #         else:
    #             return len(x.intersection(y)) / max(len(x), len(y))
    #
    #
    #     def str_eq(x: str, y: str):
    #         return x == y
    #
    #
    #     SIM_FUNC_DICT = {
    #         ("title", "jaccard"): token_ops(td.jaccard.normalized_similarity),
    #         ("title", "overlap"): token_ops(td.overlap.normalized_similarity),
    #         # ("title", "damerau_levenshtein"): td.damerau_levenshtein.normalized_similarity,
    #         ("title", "jaro_winkler"): td.jaro_winkler.normalized_similarity,
    #         # ("brand", "set_jaccard"): set_jaccard,
    #         # ("cpu_model", "set_jaccard"): set_jaccard,
    #         # ("ram_capacity", "equals"): str_eq,
    #         # ("family", "equals"): str_eq,
    #         ("pc_name", "equals"): str_eq
    #     }
    #
    #
    #     def record_sim_func(record_pair):
    #         record_left, record_right = record_pair
    #         feature_dict = {}
    #
    #         for (field, sim_func_name), sim_func in SIM_FUNC_DICT.items():
    #             x = record_left[field]
    #             y = record_right[field]
    #             if x and y:
    #                 if sim_func_name.startswith('abs_diff') and (math.isnan(x) or math.isnan(y)):
    #                     sim = -1.0
    #                 else:
    #                     sim = sim_func(x, y)
    #             else:
    #                 sim = -1.0
    #             feature_dict[f"{field}_{sim_func_name}"] = sim
    #
    #         return feature_dict
    #
    #
    #     from collections import defaultdict
    #     import multiprocessing
    #     from tqdm.auto import tqdm
    #
    #
    #     def compare_pairs(record_dict, found_pair_set):
    #         all_feature_dict = defaultdict(list)
    #         chunksize = 100
    #         tasks = (
    #             (record_dict.iloc[id_left], record_dict.iloc[id_right])
    #             # for (id_left,id_right,_,_)
    #             for (id_left, id_right, _)
    #             in found_pair_set
    #         )
    #
    #         with multiprocessing.Pool() as pool:
    #             for feature_dict in tqdm(
    #                     pool.imap(record_sim_func, tasks, chunksize=chunksize),
    #                     total=len(found_pair_set)
    #             ):
    #                 for feature, val in feature_dict.items():
    #                     all_feature_dict[feature].append(val)
    #
    #             pool.close()
    #             pool.join()
    #
    #         # found_pair_set=pd.DataFrame(found_pair_set,columns=['left_id','right_id','cosine','label'])
    #         found_pair_set = pd.DataFrame(found_pair_set, columns=['left_id', 'right_id', 'cosine'])
    #         # return pd.DataFrame(all_feature_dict, index=pd.MultiIndex.from_tuples(found_pair_set))\
    #         all_feature_dict = pd.DataFrame(all_feature_dict)
    #         return pd.concat([found_pair_set, all_feature_dict], axis=1)
    #
    #
    #     # train_pairs = compare_pairs(features, train_data)
    #     # # print(train_pairs)
    #     # # print(train_pairs.columns)
    #     # y_label = train_pairs['label']
    #     # measures = train_pairs.drop(columns=['left_id', 'right_id', 'label'], inplace=False)
    #     # from sklearn.model_selection import train_test_split
    #     #
    #     # X_train, X_test, y_train, y_test = train_test_split(measures,y_label,test_size=0.25,random_state=1234,shuffle=True)
    #     #
    #     # from sklearn.ensemble import RandomForestClassifier
    #     # from sklearn.svm import SVC
    #     # from sklearn.model_selection import PredefinedSplit, GridSearchCV
    #     # import numpy as np
    #     #
    #     # train_valid_feature_df = pd.concat([X_train, X_test])
    #     # train_valid_true_y = np.concatenate([y_train, y_test])
    #     # cv = PredefinedSplit(
    #     #     np.concatenate([
    #     #         np.full(y_train.shape[0], -1, dtype='i4'),
    #     #         np.zeros(y_test.shape[0], dtype='i4')
    #     #     ])
    #     # )
    #     # param_grid = {
    #     #     'n_estimators': [10,20,30,40],
    #     #     'max_depth': [5, 10, 25, 50],
    #     #     'min_samples_leaf': [3, 5,7],
    #     # }
    #     # # param_grid = {
    #     # #     'C': [0.1,0.5,1.0,2.0],
    #     # #     'kernel': ['linear','poly','rbf','sigmoid']
    #     # # }
    #     # clf = RandomForestClassifier(oob_score=True, random_state=1234)
    #     # # clf=SVC(probability=True)
    #     # clf = GridSearchCV(clf, param_grid, scoring='f1', cv=cv, verbose=10, n_jobs=-1)
    #     # clf.fit(train_valid_feature_df, train_valid_true_y)
    #     # import pickle
    #     # f=open("RFC_measure_jaje_2.bin",'wb')
    #     # pickle.dump(clf,f)
    #     # f.close()
    #
    #     import pickle
    #
    #     f = open("RFC_measure_jaje_2.bin", 'rb')
    #     clf = pickle.load(f)
    #     f.close()
    #     print("Best Config:")
    #     print(clf.best_params_)
    #     print("Best score:")
    #     print(clf.best_score_)
    #     # print("OOB score:")
    #     # print(clf.best_estimator_.oob_score_)
    #
    #     thres = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #     # for threshold in thres:
    #     #     print("Threshold: %f"%threshold)
    #     #     y_pred=clf.predict_proba(measures)
    #     #     y_pred[y_pred>=threshold]=1
    #     #     y_pred[y_pred<threshold]=0
    #     #     y_pred=y_pred[:,1]
    #     #     output=[]
    #     #     left=train_pairs['left_id'].values
    #     #     right=train_pairs['right_id'].values
    #     #     cosine=train_pairs['cosine'].values
    #     #     for i in range(len(y_pred)):
    #     #         if y_pred[i]==1:
    #     #             s1=ids[left[i]]
    #     #             s2=ids[right[i]]
    #     #             small=min(s1,s2)
    #     #             large=max(s1,s2)
    #     #             output.append((small,large,cosine[i]))
    #     #     print(len(output))
    #     #     print(recall_calculation(output,gnd))
    #     #     output.sort(key=lambda x: x[2])
    #     #     limit = 2814
    #     #     output = output[:limit]
    #     #     print(recall_calculation(output,gnd))
    #
    #     print("-------------------------------------------------------------------------")
    #
    #     data = pd.read_csv("./data/x1_test.csv")
    #     features = clean(data)
    #     model = SentenceTransformer(path, device='cpu')
    #     encodings = model.encode(sentences=data['title'], batch_size=256, normalize_embeddings=True)
    #     topk = 50
    #     buckets = defaultdict(list)
    #     for idx in range(data.shape[0]):
    #         brands = features['brand'][idx]
    #         for brand in brands:
    #             buckets[brand].append(idx)
    #         if len(brands) == 0:
    #             buckets['0'].append(idx)
    #     ids = data['id'].values
    #     visited_set = set()
    #     ram_capacity_list = features['ram_capacity'].values
    #     family_list = features['family'].values
    #     cpu_model_list = features['cpu_model'].values
    #     title_list = features['title'].values
    #     brand_list = features['brand'].values
    #     pcname_list = features['pc_name'].values
    #     candidate_pairs = []
    #     for key in buckets:
    #         cluster = buckets[key]
    #         embedding_matrix = encodings[cluster]
    #         k = min(topk, len(cluster))
    #         index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
    #         index_model.hnsw.efConstruction = 100
    #         index_model.add(embedding_matrix)
    #         index_model.hnsw.efSearch = 256
    #         D, I = index_model.search(embedding_matrix, k)
    #         for i in range(len(D)):
    #             for j in range(len(D[0])):
    #                 index1 = cluster[i]
    #                 index2 = cluster[I[i][j]]
    #                 s1 = ids[index1]
    #                 s2 = ids[index2]
    #                 if s1 == s2:
    #                     continue
    #                 small = min(s1, s2)
    #                 large = max(s1, s2)
    #                 visit_token = str(small) + " " + str(large)
    #                 if visit_token in visited_set:
    #                     continue
    #                 visited_set.add(visit_token)
    #                 if not (ram_capacity_list[index1] == '0' or ram_capacity_list[index2] == '0' or ram_capacity_list[
    #                     index1] == ram_capacity_list[index2]):
    #                     if family_list[index1] != 'x220' and family_list[index2] != 'x220':
    #                         continue
    #                 intersect = cpu_model_list[index1].intersection(cpu_model_list[index2])
    #                 if not (len(cpu_model_list[index1]) == 0 or len(cpu_model_list[index2]) == 0 or len(
    #                         intersect) != 0):
    #                     continue
    #                 candidate_pairs.append([index1, index2, D[i][j]])
    #
    #     train_pairs = compare_pairs(features, candidate_pairs)
    #     measures = train_pairs.drop(columns=['left_id', 'right_id'], inplace=False)
    #     gnd = pd.read_csv("./data/y1_test.csv")
    #     for threshold in thres:
    #         print("Threshold: %f" % threshold)
    #         y_pred = clf.predict_proba(measures)
    #         y_pred[y_pred >= threshold] = 1
    #         y_pred[y_pred < threshold] = 0
    #         y_pred = y_pred[:, 1]
    #         output = []
    #         left = train_pairs['left_id'].values
    #         right = train_pairs['right_id'].values
    #         cosine = train_pairs['cosine'].values
    #         for i in range(len(y_pred)):
    #             if y_pred[i] == 1:
    #                 s1 = ids[left[i]]
    #                 s2 = ids[right[i]]
    #                 small = min(s1, s2)
    #                 large = max(s1, s2)
    #                 output.append((small, large, cosine[i]))
    #         print(len(output))
    #         print(recall_calculation(output, gnd))
    #         output.sort(key=lambda x: x[2])
    #         limit = 488
    #         output = output[:limit]
    #         print(recall_calculation(output, gnd))
    #
    #     print("-------------------------------------------------------------------------")
    #
    #     data = pd.read_csv("./X1.csv")
    #     features = clean(data)
    #     model = SentenceTransformer(path, device='cpu')
    #     encodings = model.encode(sentences=data['title'], batch_size=256, normalize_embeddings=True)
    #     topk = 50
    #     buckets = defaultdict(list)
    #     for idx in range(data.shape[0]):
    #         brands = features['brand'][idx]
    #         for brand in brands:
    #             buckets[brand].append(idx)
    #         if len(brands) == 0:
    #             buckets['0'].append(idx)
    #     ids = data['id'].values
    #     visited_set = set()
    #     ram_capacity_list = features['ram_capacity'].values
    #     family_list = features['family'].values
    #     cpu_model_list = features['cpu_model'].values
    #     title_list = features['title'].values
    #     brand_list = features['brand'].values
    #     pcname_list = features['pc_name'].values
    #     candidate_pairs = []
    #     for key in buckets:
    #         cluster = buckets[key]
    #         embedding_matrix = encodings[cluster]
    #         k = min(topk, len(cluster))
    #         index_model = faiss.IndexHNSWFlat(len(embedding_matrix[0]), 8)
    #         index_model.hnsw.efConstruction = 100
    #         index_model.add(embedding_matrix)
    #         index_model.hnsw.efSearch = 256
    #         D, I = index_model.search(embedding_matrix, k)
    #         for i in range(len(D)):
    #             for j in range(len(D[0])):
    #                 index1 = cluster[i]
    #                 index2 = cluster[I[i][j]]
    #                 s1 = ids[index1]
    #                 s2 = ids[index2]
    #                 if s1 == s2:
    #                     continue
    #                 small = min(s1, s2)
    #                 large = max(s1, s2)
    #                 visit_token = str(small) + " " + str(large)
    #                 if visit_token in visited_set:
    #                     continue
    #                 visited_set.add(visit_token)
    #                 if not (ram_capacity_list[index1] == '0' or ram_capacity_list[index2] == '0' or ram_capacity_list[
    #                     index1] == ram_capacity_list[index2]):
    #                     if family_list[index1] != 'x220' and family_list[index2] != 'x220':
    #                         continue
    #                 intersect = cpu_model_list[index1].intersection(cpu_model_list[index2])
    #                 if not (len(cpu_model_list[index1]) == 0 or len(cpu_model_list[index2]) == 0 or len(
    #                         intersect) != 0):
    #                     continue
    #                 candidate_pairs.append([index1, index2, D[i][j]])
    #     # print("Start generate 2M data")
    #     import random
    #     # for i in range(1000000):
    #     #     candidate_pairs.append([random.randint(0,data.shape[0]-1),random.randint(0,data.shape[0]-1),random.random()])
    #     import time
    #
    #     print(time.time())
    #     train_pairs = compare_pairs(features, candidate_pairs)
    #     measures = train_pairs.drop(columns=['left_id', 'right_id'], inplace=False)
    #     gnd = pd.read_csv("./Y1.csv")
    #     print("Start Inference")
    #     for threshold in thres:
    #         print("Threshold: %f" % threshold)
    #         y_pred = clf.predict_proba(measures)
    #         print(time.time())
    #         print(measures.shape[0])
    #         print(y_pred)
    #         y_pred[y_pred >= threshold] = 1
    #         y_pred[y_pred < threshold] = 0
    #         y_pred = y_pred[:, 1]
    #         print(y_pred.shape[0])
    #         output = []
    #         left = train_pairs['left_id'].values
    #         right = train_pairs['right_id'].values
    #         cosine = train_pairs['cosine'].values
    #         for i in range(len(y_pred)):
    #             if y_pred[i] == 1:
    #                 s1 = ids[left[i]]
    #                 s2 = ids[right[i]]
    #                 small = min(s1, s2)
    #                 large = max(s1, s2)
    #                 output.append((small, large, cosine[i]))
    #         print(len(output))
    #         print(recall_calculation(output, gnd))
    #         output.sort(key=lambda x: x[2])
    #         limit = 2814
    #         output = output[:limit]
    #         print(recall_calculation(output, gnd))
    #         print(len(output))
    #     print(time.time())
