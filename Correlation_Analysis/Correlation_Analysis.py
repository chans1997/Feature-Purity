
import numpy as np
from pandas import DataFrame
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import matplotlib.pyplot as plot
from scipy.spatial.distance import pdist
# 设定向量长度，均为100
n = 100
file_path = 'F:/result/evaluate/FPE_CONVERAGE_DATA/R101.csv'
FP_TEST_D161 = pd.read_csv(file_path, header=0)
# corMat = DataFrame(FP_TEST_D161.corr())
# print(corMat)
# plot.pcolor(corMat)
# plot.show()
deletfive = True
if deletfive:
      n = 95
      i = 5
      x1 = FP_TEST_D161['max_0'][i:]
      print(len(x1))
      x2 = FP_TEST_D161['max_1'][i:]
      x3 = FP_TEST_D161['Acc'][i:]
      x4 = FP_TEST_D161['Kappa'][i:]
      x5 = FP_TEST_D161['AUC_Micro'][i:]
      x6 = FP_TEST_D161['AUC_Macro'][i:]
else:
      x1 = FP_TEST_D161['max_0']
      x2 = FP_TEST_D161['max_1']
      x3 = FP_TEST_D161['Acc']
      x4 = FP_TEST_D161['Kappa']
      x5 = FP_TEST_D161['AUC_Micro']
      x6 = FP_TEST_D161['AUC_Macro']

# x1 = np.random.random_integers(0, 10, (n,1))
#
# x2 = np.random.random_integers(0, 10, (n,1))
#
# x3 = np.random.random_integers(0, 10, (n,1))
# x1 = np.squeeze(x1)
# x2 = np.squeeze(x2)
# x3 = np.squeeze(x3)
print(x1, x1.shape)
# x1, x2, x3 = x1.astype('float64'), x2.astype('float64'), x2.astype('float64')
# r12, p12 = pearsonr(x1.values, x2.values)
r13, p13 = pearsonr(x1.values, x3.values)
r23, p23 = pearsonr(x2.values, x3.values)
r14, p14 = pearsonr(x1.values, x4.values)
r24, p24 = pearsonr(x2.values, x4.values)
r15, p15 = pearsonr(x1.values, x5.values)
r25, p25 = pearsonr(x2.values, x5.values)
r16, p16 = pearsonr(x1.values, x6.values)
r26, p26 = pearsonr(x2.values, x6.values)

# r12 =1 - pearsonr(np.squeeze(x1), np.squeeze(x2))[0]
# r13 =1 - pearsonr(np.squeeze(x1), np.squeeze(x3))[0]
# r23 =1 - pearsonr(np.squeeze(x2), np.squeeze(x3))[0]


d13 = (euclidean(x1.values, x3.values)**2) / (2*n)
d23 = (euclidean(x2.values, x3.values)**2) / (2*n)
d14 = (euclidean(x1.values, x4.values)**2) / (2*n)
d24 = (euclidean(x2.values, x4.values)**2) / (2*n)
d15 = (euclidean(x1.values, x5.values)**2) / (2*n)
d25 = (euclidean(x2.values, x5.values)**2) / (2*n)
d16 = (euclidean(x1.values, x6.values)**2) / (2*n)
d26 = (euclidean(x2.values, x6.values)**2) / (2*n)



c13 = cosine(x1.values, x3.values)
c23 = cosine(x2.values, x3.values)
c14 = cosine(x1.values, x4.values)
c24 = cosine(x2.values, x4.values)
c15 = cosine(x1.values, x5.values)
c25 = cosine(x2.values, x5.values)
c16 = cosine(x1.values, x6.values)
c26 = cosine(x2.values, x6.values)

sys.stdout = open('F:/result/evaluate/FPE_CONVERAGE_DATA/incep3delet1.txt', 'wt')
# f1.write('\n原始数据，没有标准化\n')
# f1.writelines('r:   ', [np.round(r12, decimals=4), np.round(r13, decimals=4),
#   np.round(r23, decimals=4)])
# f1.write('pearson:    ', np.round(p12, decimals=4), np.round(p13, decimals=4),
#   np.round(p23, decimals=4))
# f1.write('cos:        ', np.round(c12, decimals=4), np.round(c13, decimals=4),
#   np.round(c23, decimals=4))
# f1.write('euclidean sq', np.round(d12, decimals=4), np.round(d13, decimals=4),
#   np.round(d23, decimals=4))
print('\n原始数据，没有标准化\n')
print('                x1&x3  x2&x3  x1&x4  x2&x4 x1&x5  x2&x5 x1&x6  x2&x6')
print('pearsonr   :', np.round(r13, decimals=4), np.round(r23, decimals=4),np.round(r14, decimals=4), np.round(r24, decimals=4),np.round(r15, decimals=4), np.round(r25, decimals=4),np.round(r16, decimals=4), np.round(r26, decimals=4))
print('P_value    :', np.round(p13, decimals=4), np.round(p23, decimals=4),np.round(p14, decimals=4), np.round(p24, decimals=4),np.round(p15, decimals=4), np.round(p25, decimals=4),np.round(p16, decimals=4), np.round(p26, decimals=4))
print('cos        :', np.round(c13, decimals=4), np.round(c23, decimals=4),np.round(c14, decimals=4), np.round(c24, decimals=4),np.round(c15, decimals=4), np.round(c25, decimals=4),np.round(c16, decimals=4), np.round(c26, decimals=4))
print('euclidean  :', np.round(d13, decimals=4), np.round(d23, decimals=4),np.round(d14, decimals=4), np.round(d24, decimals=4),np.round(d15, decimals=4), np.round(d25, decimals=4),np.round(d16, decimals=4), np.round(d26, decimals=4))


# 标准化后的数据
x1_n = StandardScaler().fit_transform(x1.values.reshape(-1,1))
x2_n = StandardScaler().fit_transform(x2.values.reshape(-1,1))
x3_n = StandardScaler().fit_transform(x3.values.reshape(-1,1))
x4_n = StandardScaler().fit_transform(x4.values.reshape(-1,1))
x5_n = StandardScaler().fit_transform(x5.values.reshape(-1,1))
x6_n = StandardScaler().fit_transform(x6.values.reshape(-1,1))

#计算personr

p13_n = 1 - pearsonr(np.squeeze(x1_n), np.squeeze(x3_n))[0]
p23_n = 1 - pearsonr(np.squeeze(x2_n), np.squeeze(x3_n))[0]
p14_n = 1 - pearsonr(np.squeeze(x1_n), np.squeeze(x4_n))[0]
p24_n = 1 - pearsonr(np.squeeze(x2_n), np.squeeze(x4_n))[0]
p15_n = 1 - pearsonr(np.squeeze(x1_n), np.squeeze(x5_n))[0]
p25_n = 1 - pearsonr(np.squeeze(x2_n), np.squeeze(x5_n))[0]
p16_n = 1 - pearsonr(np.squeeze(x1_n), np.squeeze(x6_n))[0]
p26_n = 1 - pearsonr(np.squeeze(x2_n), np.squeeze(x6_n))[0]
# print(x1_n.shape)
# p12_n = pearsonr(np.squeeze(x1_n), np.squeeze(x2_n))[0]
# p13_n = pearsonr(np.squeeze(x1_n), np.squeeze(x3_n))[0]
# p23_n = pearsonr(np.squeeze(x2_n), np.squeeze(x3_n))[0]

#计算欧式距离
d13_n = (euclidean(x1_n, x3_n)**2) / (2*n)
d23_n = (euclidean(x2_n, x3_n)**2) / (2*n)
d14_n = (euclidean(x1_n, x4_n)**2) / (2*n)
d24_n = (euclidean(x2_n, x4_n)**2) / (2*n)
d15_n = (euclidean(x1_n, x5_n)**2) / (2*n)
d25_n = (euclidean(x2_n, x5_n)**2) / (2*n)
d16_n = (euclidean(x1_n, x6_n)**2) / (2*n)
d26_n = (euclidean(x2_n, x6_n)**2) / (2*n)


c13_n = cosine(x1_n, x3_n)
c23_n = cosine(x2_n, x3_n)
c14_n = cosine(x1_n, x4_n)
c24_n = cosine(x2_n, x4_n)
c15_n = cosine(x1_n, x5_n)
c25_n = cosine(x2_n, x5_n)
c16_n = cosine(x1_n, x6_n)
c26_n = cosine(x2_n, x6_n)


print('\n标准化后的数据: 均值=0，标准差=1\n')
# print('                x2&x3  x1&x3')
print('                x1&x3  x2&x3  x1&x4  x2&x4 x1&x5  x2&x5 x1&x6  x2&x6')
#print('pearsonr   :', np.round(r13, decimals=4), np.round(r23, decimals=4),np.round(r14, decimals=4), np.round(r24, decimals=4),np.round(r15, decimals=4), np.round(r25, decimals=4),np.round(r16, decimals=4), np.round(r26, decimals=4))
print('pearsonr    :', np.round(p13_n, decimals=4), np.round(p23_n, decimals=4),np.round(p14_n, decimals=4), np.round(p24_n, decimals=4),np.round(p15_n, decimals=4), np.round(p25_n, decimals=4),np.round(p16_n, decimals=4), np.round(p26_n, decimals=4))
print('cos        :', np.round(c13_n, decimals=4), np.round(c23_n, decimals=4),np.round(c14_n, decimals=4), np.round(c24_n, decimals=4),np.round(c15_n, decimals=4), np.round(c25_n, decimals=4),np.round(c16_n, decimals=4), np.round(c26_n, decimals=4))
print('euclidean  :', np.round(d13_n, decimals=4), np.round(d23_n, decimals=4),np.round(d14_n, decimals=4), np.round(d24_n, decimals=4),np.round(d15_n, decimals=4), np.round(d25_n, decimals=4),np.round(d16_n, decimals=4), np.round(d26_n, decimals=4))
