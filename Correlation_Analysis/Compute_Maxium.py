import numpy as np
from scipy.signal import argrelextrema
from pandas import DataFrame
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import os
import glob

def Compute_Maxium(model, ACC_Max_L=[], Max_0_FP_acc=[], Max_1_FP_acc=[], Max_0_FP_error=[],
                   Max_1_FP_error=[], ACC_greater_num=[], Max_0_FP_greater_num=[],
                   Max_1_FP_greater_num=[],Max_0_FP_greater_num_error=[], Max_1_FP_greater_num_error =[],
                   Max_0_FP_greater_num_max=[], Max_1_FP_greater_num_max=[]):
    file_path = 'F:/result/evaluate/FPE_CONVERAGE_DATA/{}.csv'.format(model)
    FP_TEST_D161 = pd.read_csv(file_path, header=0)

    x1 = FP_TEST_D161['max_0']
    x2 = FP_TEST_D161['max_1']
    x3 = FP_TEST_D161['AUC_Macro']

    max_0 = argrelextrema(x1.values, np.greater)
    max_1 = argrelextrema(x2.values, np.greater)
    max_acc = argrelextrema(x3.values, np.greater)
    # print(x1)
    # print(x2)
    # print(max_0, max_0[0].shape)
    # print(max_1, max_1[0].shape)
    # print(max_acc, max_acc[0].shape)

    print(x3.max())
    ACC_Max_L.append(x3.max())

    print('test', FP_TEST_D161['max_0'].argmax())
    print('test', FP_TEST_D161['max_1'].argmax())
    argmax_0 = FP_TEST_D161['max_0'].argmax()
    argmax_1 = FP_TEST_D161['max_1'].argmax()

    print(argmax_0, x3[argmax_0])
    print(argmax_1, x3[argmax_1])
    Max_0_FP_acc.append(x3[argmax_0])
    Max_1_FP_acc.append(x3[argmax_1])

    max_0_error = (x3.max()- x3[argmax_0])/x3.max()
    max_1_error = (x3.max() - x3[argmax_1]) / x3.max()
    print(max_0_error)
    print(max_1_error)
    Max_0_FP_error.append('{}'.format(max_0_error,'.4f'))
    Max_1_FP_error.append('{}'.format(max_1_error,'.4f'))

    print(max_acc[0].shape)
    print(max_0[0].shape)
    print(max_1[0].shape)
    ACC_greater_num.append(max_acc[0].shape)
    Max_0_FP_greater_num.append(max_0[0].shape)
    Max_1_FP_greater_num.append(max_1[0].shape)
                                  #acc
    max0_FP_gre_error = (x3.max()-x3.values[max_0].max())/x3.max()
    Max_0_FP_greater_num_error.append('{}'.format(max0_FP_gre_error, '.4f'))
    print(max0_FP_gre_error)
    max1_FP_gre_error = (x3.max() - x3.values[max_1].max()) / x3.max()
    Max_1_FP_greater_num_error.append('{}'.format(max1_FP_gre_error, '.4f'))
    print(max0_FP_gre_error)

    # print(FP_TEST_D161['Acc'].values[max_0].shape)
    Max_0_FP_greater_num_max.append(x3.values[max_0].max())
    Max_1_FP_greater_num_max.append(x3.values[max_1].max())

    print(x3.values[max_0].max())
    print(x3.values[max_1].max())
    print(max0_FP_gre_error)
    print(max1_FP_gre_error)


def Compute_Maxium_Delete5(model, ACC_Max_L=[], Max_0_FP_acc=[], Max_1_FP_acc=[], Max_0_FP_error=[],
                   Max_1_FP_error=[], ACC_greater_num=[], Max_0_FP_greater_num=[],
                   Max_1_FP_greater_num=[],Max_0_FP_greater_num_error=[], Max_1_FP_greater_num_error =[],
                   Max_0_FP_greater_num_max=[], Max_1_FP_greater_num_max=[]):
    file_path = 'F:/result/evaluate/FPE_CONVERAGE_DATA/{}.csv'.format(model)
    FP_TEST_D161 = pd.read_csv(file_path, header=0)

    x0 = FP_TEST_D161['max_0'][90:]
    x1 = FP_TEST_D161['max_1'][90:]
    x3 = FP_TEST_D161['AUC_Micro'][90:]

    max_0 = argrelextrema(x0.values, np.greater)
    # max_0 = (max_0[0]+5)
    max_1 = argrelextrema(x1.values, np.greater)
    # max_1 = (max_1[0] + 5)
    max_acc = argrelextrema(x3.values, np.greater)
    # print(x1)
    # print(x2)
    # print(max_0, max_0[0].shape)
    # print(max_1, max_1[0].shape)
    # print(max_acc, max_acc[0].shape)

    print(x3.max())
    ACC_Max_L.append(x3.max())

    print('test', x0.argmax())
    print('test', x1.argmax())
    argmax_0 = x0.argmax()
    argmax_1 = x1.argmax()

    print(argmax_0, x3.values[argmax_0])
    print(argmax_1, x3.values[argmax_1])
    Max_0_FP_acc.append(x3.values[argmax_0])
    Max_1_FP_acc.append(x3.values[argmax_1])

    max_0_error = (x3.max()- x3.values[argmax_0])/x3.max()
    max_1_error = (x3.max() - x3.values[argmax_1]) / x3.max()
    print(max_0_error)
    print(max_1_error)
    Max_0_FP_error.append('{}'.format(max_0_error,'.4f'))
    Max_1_FP_error.append('{}'.format(max_1_error,'.4f'))

    print(max_acc[0].shape)
    print(max_0[0].shape)
    print(max_1[0].shape)
    ACC_greater_num.append(max_acc[0].shape)
    Max_0_FP_greater_num.append(max_0[0].shape)
    Max_1_FP_greater_num.append(max_1[0].shape)

    max0_FP_gre_error = (x3.max()-x3.values[max_0].max())/x3.max()
    Max_0_FP_greater_num_error.append('{}'.format(max0_FP_gre_error, '.4f'))
    print(max0_FP_gre_error)
    max1_FP_gre_error = (x3.max() - x3.values[max_1].max()) / x3.max()
    Max_1_FP_greater_num_error.append('{}'.format(max1_FP_gre_error, '.4f'))
    print(max0_FP_gre_error)

    # print(FP_TEST_D161['Acc'].values[max_0].shape)
    Max_0_FP_greater_num_max.append(x3.values[max_0].max())
    Max_1_FP_greater_num_max.append(x3.values[max_1].max())

    print(x3.values[max_0].max())
    print(x3.values[max_1].max())
    print(max0_FP_gre_error)
    print(max1_FP_gre_error)
# for local minima
# z = argrelextrema(x, np.less)


# print('误差:' )

if __name__ == '__main__':

    models = glob.glob(os.path.join("F:/result/evaluate/FPE_CONVERAGE_DATA/", "*.csv"))
    models = [model.split('\\')[1] for model in models]
    model = [model.split('.')[0] for model in models]
    eval_csv_file = 'F:/result/evaluate/FPEval_Error_Compute/AUC_Micro_Error_Compute_Delete90.csv'
    ACC_Max_L = []
    Max_0_FP_acc = []
    Max_1_FP_acc = []
    Max_0_FP_error = []
    Max_1_FP_error = []
    ACC_greater_num = []
    Max_0_FP_greater_num = []
    Max_1_FP_greater_num = []
    Max_0_FP_greater_num_error = []
    Max_1_FP_greater_num_error = []
    Max_0_FP_greater_num_max = []
    Max_1_FP_greater_num_max = []
    for i in model:
        Compute_Maxium_Delete5(i, ACC_Max_L, Max_0_FP_acc, Max_1_FP_acc, Max_0_FP_error,
                   Max_1_FP_error, ACC_greater_num, Max_0_FP_greater_num,
                   Max_1_FP_greater_num, Max_0_FP_greater_num_error, Max_1_FP_greater_num_error,
                   Max_0_FP_greater_num_max, Max_1_FP_greater_num_max)

    model_column = pd.Series(model, name='model')
    ACC_Max_L_column = pd.Series(ACC_Max_L, name='Best_ACC')

    Max_0_FP_acc_column = pd.Series(Max_0_FP_acc, name='Max_0_FP_acc')
    Max_1_FP_acc_column = pd.Series(Max_1_FP_acc, name='Max_1_FP_acc')

    Max_0_FP_error_column = pd.Series(Max_0_FP_error, name='Max_0_FP_error')
    Max_1FP_error_column = pd.Series(Max_1_FP_error, name='Max_1_FP_error')

    ACC_greater_num_column = pd.Series(ACC_greater_num, name='ACC_greater_num')
    Max_0_FP_greater_column = pd.Series(Max_0_FP_greater_num, name='Max_0_FP_greater_num')
    Max_1_FP_greater_column = pd.Series(Max_1_FP_greater_num, name='Max_1_FP_greater_num')

    Max_0_FP_greater_num_error_column = pd.Series(Max_0_FP_greater_num_error, name='Max_0_FP_greater_num_error')
    Max_1_FP_greater_num_error_column = pd.Series(Max_1_FP_greater_num_error, name='Max_1_FP_greater_num_error')

    Max_0_FP_greater_num_max_column = pd.Series(Max_0_FP_greater_num_max, name='Max_0_FP_greater_num_max')
    Max_1_FP_greater_num_max_column = pd.Series(Max_1_FP_greater_num_max, name='Max_1_FP_greater_num_max')

    label_total = pd.concat([model_column, ACC_Max_L_column, Max_0_FP_acc_column, Max_1_FP_acc_column, Max_0_FP_error_column,
                             Max_1FP_error_column, ACC_greater_num_column, Max_0_FP_greater_column, Max_1_FP_greater_column, Max_0_FP_greater_num_error_column,
                             Max_1_FP_greater_num_error_column, Max_0_FP_greater_num_max_column, Max_0_FP_greater_num_max_column], axis=1)
    save = pd.DataFrame(label_total)
    save.to_csv(eval_csv_file, index=True, sep=',')






