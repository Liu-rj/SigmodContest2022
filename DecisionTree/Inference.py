import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
import joblib
import pickle

def infer_x1(csv_path,feature_path,clf_path,gnd_file=None):
    x1=pd.read_csv(csv_path)
    datalist = x1.values.tolist()
    id_feature={}
    length=len(datalist)
    for i in range(length):
        rid=datalist[i][0]
        rtitle=datalist[i][1].lower().split()
        record_feature=set()
        for string in rtitle:
            if len(string) == 1 and not string.isdigit():
                continue
            record_feature.add(string)
        id_feature[str(rid)]=record_feature

    feaure_file=open(feature_path,'rb')
    feature_list=pickle.load(feaure_file)
    feaure_file.close()

    binary_data = pd.DataFrame()
    binary_data['id'] = x1['id']
    for feat in feature_list:
        binary_value = []
        for i in range(length):
            feature_of_id = id_feature[str(datalist[i][0])]
            if feat in feature_of_id:
                binary_value.append(1)
            else:
                binary_value.append(0)
        binary_data[feat] = binary_value

    X = binary_data.drop(['id'], axis=1)
    clf = joblib.load(clf_path)
    y_predict=clf.predict(X)
    max_y=max(y_predict)
    cluster_list=[None]
    for i in range(max_y):
        cluster_list.append([])
    for i in range(length):
        cluster_list[y_predict[i]].append(datalist[i][0])
    left_list=[]
    right_list=[]
    size=0
    for i in range(1,max_y+1):
        current_cluster=cluster_list[i]
        for j in range(0,len(current_cluster)-1):
            for k in range(j+1,len(current_cluster)):
                if size<1000000:
                    left_id=min(current_cluster[j],current_cluster[k])
                    right_id=max(current_cluster[j],current_cluster[k])
                    left_list.append(left_id)
                    right_list.append(right_id)
                    size+=1

    if gnd_file is not None:
        gnd = pd.read_csv(gnd_file)
        cnt = 0
        for i in range(len(left_list)):
            if not gnd[(gnd['lid'] == left_list[i]) & (gnd['rid'] == right_list[i])].empty:
                cnt += 1
        print("Recall for X1 is %f"%(cnt/gnd.values.shape[0]))

    while size<1000000:
        left_list.append(0)
        right_list.append(0)
        size+=1

    return left_list,right_list

def infer_x2(csv_path,feature_path,clf_path,gnd_file=None):
    x1 = pd.read_csv(csv_path)
    datalist = x1[['id','name','brand','description','category']].values.tolist()
    id_feature = {}
    length = len(datalist)
    for i in range(length):
        record=datalist[i]
        rid = record[0]
        record_feature = set()
        rname = record[1].lower().split()
        for string in rname:
            if len(string) == 1 and not string.isdigit():
                continue
            record_feature.add(string)
        if type(record[2]) == type("a"):
            rname = record[2].lower().split()
            for string in rname:
                if len(string) == 1 and not string.isdigit():
                    continue
                record_feature.add(string)
        if type(record[3]) == type("a"):
            rname = record[3].lower().split()
            for string in rname:
                if len(string) == 1 and not string.isdigit():
                    continue
                record_feature.add(string)
        if type(record[4]) == type("a"):
            rname = record[4].lower().split()
            for string in rname:
                if len(string) == 1 and not string.isdigit():
                    continue
                record_feature.add(string)
        id_feature[str(rid)] = record_feature

    feaure_file = open(feature_path, 'rb')
    feature_list = pickle.load(feaure_file)
    feaure_file.close()

    binary_data = pd.DataFrame()
    binary_data['id'] = x1['id']
    for feat in feature_list:
        binary_value = []
        for i in range(length):
            feature_of_id = id_feature[str(datalist[i][0])]
            if feat in feature_of_id:
                binary_value.append(1)
            else:
                binary_value.append(0)
        binary_data[feat] = binary_value

    binary_data['instance_price'] = x1['price']
    binary_data = binary_data.fillna(binary_data.mean())
    X = binary_data.drop(['id'], axis=1)
    clf = joblib.load(clf_path)
    y_predict = clf.predict(X)
    max_y = max(y_predict)
    cluster_list = [None]
    for i in range(max_y):
        cluster_list.append([])
    for i in range(length):
        cluster_list[y_predict[i]].append(datalist[i][0])
    left_list = []
    right_list = []
    size = 0
    for i in range(1, max_y + 1):
        current_cluster = cluster_list[i]
        for j in range(0, len(current_cluster) - 1):
            for k in range(j + 1, len(current_cluster)):
                if size < 2000000:
                    left_id = min(current_cluster[j], current_cluster[k])
                    right_id = max(current_cluster[j], current_cluster[k])
                    left_list.append(left_id)
                    right_list.append(right_id)
                    size += 1

    if gnd_file is not None:
        gnd = pd.read_csv(gnd_file)
        cnt = 0
        for i in range(len(left_list)):
            if not gnd[(gnd['lid'] == left_list[i]) & (gnd['rid'] == right_list[i])].empty:
                cnt += 1
        print("Recall for X2 is %f"%(cnt/gnd.values.shape[0]))

    while size < 2000000:
        left_list.append(0)
        right_list.append(0)
        size += 1

    return left_list, right_list

if __name__=='__main__':
    x1_left,x1_right=infer_x1("../X1.csv",'feature_list.txt','simpleDecisionTree.bin','../Y1.csv')
    x2_left,x2_right=infer_x2("../X2.csv",'x2_feature_list.txt','x2_simpleDecisionTree.bin','../Y2.csv')
    left_list=x1_left+x2_left
    right_list=x1_right+x2_right
    output=pd.DataFrame()
    output['left_instance_id']=left_list
    output['right_instance_id']=right_list
    output.to_csv("output.csv",index=False)