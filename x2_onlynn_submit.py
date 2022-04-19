import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import time
import faiss
import queue
import numpy as np
import re

def save_output(X1_candidate_pairs,
                X2_candidate_pairs):  # save the candset for both datasets to a SINGLE file output.csv
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

ram_compiled=re.compile('[\W]\d+[\W]*gb[\W]*ram|[\W]\d+[\W]*gb[\W]*memory')
memory_compiled=re.compile('\d+')
cpu_compiled=re.compile('[\W]i\d{1,2}[\W]\d{3,5}[a-z][^b]')
i_compiled=re.compile('i\d{1,2}')
cpumodel_compiled=re.compile('\d{3,5}[a-z]')
acer_model_compiled=re.compile('\W[a-z]\d-\w+-\w+\W')
acer_important_part_compiled=re.compile('[a-z]\d-[0-9]+')
elitebook_suffix_compiled=re.compile('\Welitebook e*\d+p*\W')
elitebook_model_compiled=re.compile('e*\d+p*')
tablet_suffix_compiled=re.compile('tablet \d+')
def extract_x1(t:str) -> list:
    t_feature=[]
    t_RAM=ram_compiled.search(t)
    if t_RAM is None:
        t_feature.append(None)
    else:
        str1=t_RAM.group()
        t_memory=memory_compiled.search(str1).group()
        t_feature.append(t_memory)
    t_cpu=cpu_compiled.search(t)
    if t_cpu is None:
        t_feature.append(None)
        t_feature.append(None)
    else:
        str1=t_cpu.group()
        t_i=i_compiled.search(str1).group()
        t_feature.append(t_i)
        t_model=cpumodel_compiled.search(str1).group()
        t_feature.append(t_model)
    if 'acer' in t:
        t_brand_model=acer_model_compiled.search(t)
        if t_brand_model is None:
            t_feature.append(None)
        else:
            str1=t_brand_model.group()
            t_brand_model_id=acer_important_part_compiled.search(str1)
            if t_brand_model_id is not None:
                t_feature.append(t_brand_model_id.group())
            else:
                t_feature.append(None)
    elif 'elitebook' in t:
        t_brand_model=elitebook_suffix_compiled.search(t)
        if t_brand_model is None:
            t_feature.append(None)
        else:
            str1=t_brand_model.group()
            t_brand_model_id=elitebook_model_compiled.search(str1)
            t_feature.append(t_brand_model_id.group())
    elif 'tablet' in t:
        t_brand_model=tablet_suffix_compiled.search(t)
        if t_brand_model is None:
            t_feature.append(None)
        else:
            t_feature.append(t_brand_model.group())
    else:
        t_feature.append(None)
    return t_feature

def x1_filter(t1:list,t2:list)-> bool:
    for i in range(4):
        if t1[i] is not None and t2[i] is not None and t1[i]!=t2[i]:
            return False
    return True


def x1():
    topk=50
    limit=1000000
    data=pd.read_csv("X1.csv")
    sentences=list(map(lambda x:x.lower(),data['title']))
    model_path="fromstart_further_x1_berttiny_finetune_epoch20_margin0.01"
    model=SentenceTransformer(model_path,device='cpu')
    numpy_embedding=model.encode(sentences,batch_size=256,convert_to_numpy=True,normalize_embeddings=True)
    index=faiss.IndexHNSWFlat(len(numpy_embedding[0]),8)
    index.hnsw.efConstruction=100
    index.add(numpy_embedding)
    index.hnsw.efSearch=256
    D,I=index.search(numpy_embedding,topk)
    shape0=len(D)
    shape1=len(D[0])
    print(shape0)
    hashset={}
    maxscore=0
    features=list(map(lambda x:extract_x1(x),sentences))
    pq=queue.PriorityQueue()
    for j in range(shape1):
        flag=True
        for i in range(shape0):
            s1=min(I[i][j],i)
            s2=max(I[i][j],i)
            token=str(s1)+" "+str(s2)
            if s1==s2 or token in hashset:
                continue
            hashset[token]=1
            if pq.qsize()<limit:
                flag=False
                if not x1_filter(features[s1],features[s2]):
                    continue
                pq.put((-D[i][j],s1,s2))
                maxscore=max(maxscore,D[i][j])
            elif D[i][j]<maxscore:
                flag=False
                if not x1_filter(features[s1],features[s2]):
                    continue
                pq.put((-D[i][j],s1,s2))
                entry=pq.get()
                maxscore=-entry[0]
        if flag:
            break
    output=[]
    while not pq.empty():
        score,i,j=pq.get()
        lid=data['id'][i]
        rid=data['id'][j]
        output.append((min(lid,rid),max(lid,rid)))
        if len(output)==limit:
            break
    return output

def x2():
    topk=50
    limit=2000000
    data=pd.read_csv("X2.csv")
    sentences=list(map(lambda x:x.lower(),data['name']))
    model_path="model/checkpoints/sts-mix_base/62000"
    model=SentenceTransformer(model_path,device='cpu')
    numpy_embedding=model.encode(sentences,batch_size=256,convert_to_numpy=True,normalize_embeddings=True)
    index=faiss.IndexHNSWFlat(len(numpy_embedding[0]),8)
    index.hnsw.efConstruction=100
    index.add(numpy_embedding)
    index.hnsw.efSearch=256
    D,I=index.search(numpy_embedding,topk)
    shape0=len(D)
    shape1=len(D[0])
    print(shape0)
    hashset={}
    maxscore=0
    pq=queue.PriorityQueue()
    for j in range(shape1):
        flag=True
        for i in range(shape0):
            s1=min(I[i][j],i)
            s2=max(I[i][j],i)
            token=str(s1)+" "+str(s2)
            if s1==s2 or token in hashset:
                continue
            hashset[token]=1
            if pq.qsize()<limit:
                flag=False
                pq.put((-D[i][j],s1,s2))
                maxscore=max(maxscore,D[i][j])
            elif D[i][j]<maxscore:
                flag=False
                pq.put((-D[i][j],s1,s2))
                entry=pq.get()
                maxscore=-entry[0]
        if flag:
            break
    output=[]
    while not pq.empty():
        score,i,j=pq.get()
        lid=data['id'][i]
        rid=data['id'][j]
        output.append((min(lid,rid),max(lid,rid)))
        if len(output)==limit:
            break
    return output

print(time.time())
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
x1_pair=[]
print(time.time())
dd=pd.read_csv("X1.csv")
x2_pair=x2()
save_output(x1_pair,x2_pair)
print(time.time())