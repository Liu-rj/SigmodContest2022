from typing import *

nonsense = ['|', ',', '-', ':', '/', '+', '&']


def bucket_blocking(data) -> List[Tuple[Set[str], List[int]]]:
    buckets: List[Tuple[Set[str], List[int]]] = []
    for i in range(data.shape[0]):
        max_similarity = 0
        target_bucket = None
        vector = set([it for it in data['title'][i].lower().split() if it not in nonsense])
        for bucket in buckets:
            jaccard_similarity = len(vector.intersection(bucket[0])) / max(len(vector), len(bucket[0]))
            if jaccard_similarity > max_similarity and jaccard_similarity >= 0.5:
                max_similarity = jaccard_similarity
                target_bucket = bucket
        if max_similarity > 0:
            target_bucket[1].append(data['id'][i])
        else:
            buckets.append((vector, [data['id'][i]]))
    return buckets
