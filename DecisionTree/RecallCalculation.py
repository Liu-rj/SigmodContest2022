import pandas as pd

if __name__=='__main__':
    predict_pd=pd.read_csv('output.csv')
    gnd=pd.read_csv('../Y1.csv')

    predict=predict_pd.values.tolist()

    print(gnd)
    print(predict)
    print(gnd[(gnd['lid']==predict[0][0]) & (gnd['rid']==predict[0][1])].empty)
    print(gnd.values.shape[0])
    cnt=0
    for i in range(len(predict)):
        if not gnd[(gnd['lid']==predict[i][0]) & (gnd['rid']==predict[i][1])].empty:
            cnt+=1
    print(cnt)
    print(cnt/gnd.values.shape[0])