import importlib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pickle


if __name__=='__main__':
    inference_module=importlib.import_module('DecisionTree.Inference')
    x1_left,x1_right=inference_module.infer_x1("X1.csv","./DecisionTree/feature_list.txt","./DecisionTree/simpleDecisionTree.bin",None)
    x2_left,x2_right=inference_module.infer_x2("X2.csv","./DecisionTree/x2_feature_list.txt","./DecisionTree/x2_simpleDecisionTree.bin",None)
    left_list=x1_left+x2_left
    right_list=x1_right+x2_right
    output=pd.DataFrame()
    output['left_instance_id']=left_list
    output['right_instance_id']=right_list
    output.to_csv("output.csv",index=False)