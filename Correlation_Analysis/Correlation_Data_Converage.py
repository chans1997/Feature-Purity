import pandas as pd
import os
ACC_DATA_ROOT ="F:/result/evaluate/FPEval"
ROOT = "F:/result/evaluate/FPE_CONVERAGE_DATA"
MAX1_ROOT = 'F:/result/evaluate/FPEval_MAX_1'
MAX0_ROOT = 'F:/result/evaluate/FPEval_MAX_0'

model = os.listdir(ACC_DATA_ROOT)

def ConData(model):
    FPEval_MAX0 = pd.read_csv(MAX0_ROOT+'/{}/FPEval_MAX_0_{}_fpresult.csv'.format(model, model))
    FPEval_MAX1 = pd.read_csv(MAX1_ROOT + '/{}/FPEval_MAX_1_{}_fpresult.csv'.format(model, model))
    ACC = pd.read_csv(ACC_DATA_ROOT + '/{}/eval_result.csv'.format(model, model))
    FPEval_MAX1['max_0'] = FPEval_MAX0['阈值=0.5的特征纯度']
    FPEval_MAX1['max_1'] = FPEval_MAX1['阈值=0.5的特征纯度']
    # FPEval_MAX0.rename({'阈值=0.5的特征纯度': 'max_0'}, level=0)
    # FPEval_MAX1.rename({'阈值=0.5的特征纯度': 'max_1'}, level=0)
    # print(FPEval_MAX1['max_0'])
    # data1 = FPEval_MAX0['max_0']
    # data2 = FPEval_MAX1['max_1']
    # print(data2)
    data3 = ACC
    df = pd.concat([data3, FPEval_MAX1['max_0'], FPEval_MAX1['max_1']], axis=1, ignore_index=False)
    df.to_csv(ROOT+'/{}.csv'.format(model))
    print(df)

if __name__ == '__main__':
    for i in model:
        if i !='nasnet':
            ConData(i)




