import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pickle

if __name__=='__main__':
    x1=pd.read_csv("../X1.csv")
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

    feaure_file=open('feature_list.txt','rb')
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
    clf = joblib.load('simpleDecisionTree.bin')
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
    output=pd.DataFrame()
    output['lid']=left_list
    output['rid']=right_list
    output.to_csv("output.csv",index=False)