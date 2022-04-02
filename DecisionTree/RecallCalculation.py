import pandas as pd

if __name__ == '__main__':
    predict_pd = pd.read_csv('../FeatureExtraction/hierarchical2.csv')
    #predict_pd = pd.read_csv('../output.csv')
    gnd = pd.read_csv('../Y1.csv')

    predict = predict_pd.values.tolist()
    print(len(predict))

    cnt = 0
    for i in range(min(len(predict),5000)):
        if not gnd[(gnd['lid'] == predict[i][0]) & (gnd['rid'] == predict[i][1])].empty:
            cnt += 1
    print(cnt)
    print(cnt / gnd.values.shape[0])

    cnt=0

    gnd = pd.read_csv('../Y2.csv')
    for i in range(1000000,len(predict)):
        if not gnd[(gnd['lid'] == predict[i][0]) & (gnd['rid'] == predict[i][1])].empty:
            cnt += 1
        if predict[i][0]==0 and predict[i][1]==0:
            break
    print(cnt)
    print(cnt / gnd.values.shape[0])