import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent

    def findRoot(self):
        p = self
        while p.parent is not None:
            p = p.parent
        return p

    def unionAnotherTree(self, q):
        qroot = q.findRoot()
        proot = self.findRoot()
        if proot != qroot:
            qroot.parent = proot

if __name__=="__main__":
    '''
    First part: group instances in X into clusters,
    and sort them with cluster id
    '''
    x1=pd.read_csv(r"../X1.csv",encoding="utf-8")
    y1=pd.read_csv(r"../Y1.csv",encoding="utf-8")
    lid=y1['lid'].to_list()
    rid=y1['rid'].to_list()
    id_node_map = {}
    length = len(lid)
    for i in range(length):
        leftNode = None
        rightNode = None
        try:
            leftNode = id_node_map[str(lid[i])]
        except:
            leftNode = Node(lid[i])
            id_node_map[str(lid[i])] = leftNode
        try:
            rightNode = id_node_map[str(rid[i])]
        except:
            rightNode = Node(rid[i])
            id_node_map[str(rid[i])] = rightNode
        leftNode.unionAnotherTree(rightNode)
    allNodes = {}
    ids = x1['id'].to_list()
    max_id = 1
    cluster_list=[]
    invalid_group=1
    for i in ids:
        try:
             currentNode = id_node_map[str(i)]
        except:
             currentNode=Node(i)
             invalid_group+=1
        rootNode=currentNode.findRoot()
        if str(rootNode.value) not in allNodes:
            allNodes[str(rootNode.value)]=max_id
            cluster_list.append(max_id)
            max_id+=1
        else:
            cluster_list.append(allNodes[str(rootNode.value)])
    print(max_id)
    print(invalid_group)
    x1['cluster']=cluster_list
    x1.sort_values(by=['cluster'], inplace=True)
    '''
    Second part: feature selection
    '''

    id_feature = {}
    cluster_feature = [None]
    all_feature_set = set()
    for i in range(1, 719):
        cluster_instances = x1[x1['cluster'] == i][['id', 'title']].values.tolist()
        current_cluster_set = set()
        first_flag = True
        for record in cluster_instances:
            rid = record[0]
            rtitle = record[1].lower().split()
            # calculate feature for an instance
            record_feature = set()
            for string in rtitle:
                if len(string) == 1 and not string.isdigit():
                    continue
                record_feature.add(string)
            id_feature[str(rid)] = record_feature
            #merge instance feature to cluster feature
            if first_flag:
                first_flag = False
                current_cluster_set = current_cluster_set.union(record_feature)
            else:
                current_cluster_set = current_cluster_set.intersection(record_feature)
        cluster_feature.append(current_cluster_set)
        #merge cluster feature to all the features of the dataset
        all_feature_set = all_feature_set.union(current_cluster_set)

    datalist = x1.values.tolist()
    feature_list = list(all_feature_set)
    binary_data = pd.DataFrame()
    binary_data['y'] = x1['cluster']
    binary_data['id'] = x1['id']
    for feat in feature_list:
        binary_value = []
        for i in range(len(datalist)):
            feature_of_id = id_feature[str(datalist[i][0])]
            if feat in feature_of_id:
                binary_value.append(1)
            else:
                binary_value.append(0)
        binary_data[feat] = binary_value

    '''
    Third part: Train decision tree
    '''
    X = binary_data.drop(['y', 'id'], axis=1)
    Y = binary_data['y']
    clf = DTC()
    clf = clf.fit(X, Y)
    y_predict = clf.predict(X)
    print(accuracy_score(Y,y_predict))

    '''
    (Optional): save the model
    '''
    from sklearn.externals import joblib
    joblib.dump(clf, 'simpleDecisionTree.bin')

    import pickle
    feature_file = open("feature_list.txt", 'wb')
    pickle.dump(feature_list,feature_file)
    feature_file.close()