import pandas as pd

if __name__ == '__main__':
    predict_pd = pd.read_csv('../FeatureExtraction/hierarchical.csv')
    #predict_pd = pd.read_csv('../output.csv')
    gnd = pd.read_csv('../Y1.csv')

    predict = predict_pd.values.tolist()

    cnt = 0
    for i in range(min(len(predict),1000000)):
        if not gnd[(gnd['lid'] == predict[i][0]) & (gnd['rid'] == predict[i][1])].empty:
            cnt += 1
    print(cnt)
    print(cnt / gnd.values.shape[0])
