import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize



train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train = train.set_index('sku')
test = test.set_index('sku')


float_cols = []
object_cols = []

# for c, dtype in zip(train.columns, train.dtypes):
# 	if dtype== np.float64:
# 		float_cols.append(c)
# 		print(max(train[c]), '\t', c)
# print()


# plt.figure(1)
# train.lead_time.hist()
# plt.axvline(train.lead_time.median(), 
#             color="green", 
#             linewidth=2.0)
# plt.axvline(train.lead_time.mean(), 
#             color="red", 
#             linestyle='dashed', 
#             linewidth=2.0)
# plt.show()
# print("Median of lead_time : {}".format(train.lead_time.median()))
# print("Mean of lead_time : {}".format(train.lead_time.mean()))



# plt.figure(1)
# train.perf_6_month_avg.hist()
# plt.axvline(train.perf_6_month_avg.median(), 
#             color="green", 
#             linewidth=2.0)
# plt.axvline(train.perf_6_month_avg.mean(), 
#             color="red", 
#             linestyle='dashed', 
#             linewidth=2.0)
# plt.show()
# print("Median of perf_6_month_avg : {}".format(train.perf_6_month_avg.median()))
# print("Mean of perf_6_month_avg : {}".format(train.perf_6_month_avg.mean()))


# plt.figure(2)
# train.perf_12_month_avg.hist()
# plt.axvline(train.perf_12_month_avg.median(), 
#             color="green", 
#             linewidth=2.0)
# plt.axvline(train.perf_12_month_avg.mean(), 
#             color="red", 
#             linestyle='dashed', 
#             linewidth=2.0)
# plt.show()
# print("Median of perf_12_month_avg : {}".format(train.perf_12_month_avg.median()))
# print("Mean of perf_12_month_avg : {}".format(train.perf_12_month_avg.mean()))

# choose median to imput missing values in 'lead_time'
train['lead_time'] = Imputer(strategy='median').fit_transform(train['lead_time'].values.reshape(-1,1))

## test dataset's lead_time median is same with train set
test['lead_time'] = Imputer(strategy='median').fit_transform(
    test['lead_time'].values.reshape(-1,1))


# drop NaN values
train = train.dropna()
test = test.dropna()

# choose mean to imput missing values in 'perf_month_avg'
for col in ['perf_6_month_avg', 'perf_12_month_avg']:
        train[col] = Imputer(missing_values=-99).fit_transform(
                                    train[col].values.reshape(-1, 1))

## imput values in test set with same values in train set
f6 = lambda x: -6.8720588378183045 if x == -99 else x
test["perf_6_month_avg"] = test["perf_6_month_avg"].apply(f6)
f12 = lambda x: -6.437946743213299 if x == -99 else x
test["perf_12_month_avg"] = test["perf_12_month_avg"].apply(f12)


for i in object_cols:
	train[i] = (train[i] == 'Yes').astype(int)
	test[i]=(test[i] == 'Yes').astype(int)

# print("Items not on backorder, Class 0: {}".format(train.went_on_backorder.value_counts()[0]))
# print("Items went on backorder, Class 1: {}".format(train.went_on_backorder.value_counts()[1]))

# print("Proportion of items not on backorder: {}%".format(100*train.went_on_backorder.value_counts()[0]/len(train.went_on_backorder)))
# print("Proportion of items went on backorder: {}%".format(100*train.went_on_backorder.value_counts()[1]/len(train.went_on_backorder)))
# train.went_on_backorder.hist()
# # plt.show()
# print("Items not on backorder, Class 0: {}".format(test.went_on_backorder.value_counts()[0]))
# print("Items went on backorder, Class 1: {}".format(test.went_on_backorder.value_counts()[1]))

# visualization of variables
# train.hist(figsize=(12,12), alpha=0.8, grid=False)
# plt.show()

df = pd.concat([train, test])
# print(df.shape)
# df.tail()

print(pd.DataFrame(df, index = train.index).shape)
print(pd.DataFrame(df, index = test.index).shape)



# qty_cols = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
#             'forecast_6_month', 'forecast_9_month', 'min_bank', 
#             'local_bo_qty', 'pieces_past_due', 'sales_1_month', 
#             'sales_3_month', 'sales_6_month', 'sales_9_month']

# df[qty_cols] = normalize(df[qty_cols], axis=1)
# df[qty_cols].hist(figsize=(10,11), color='orange', grid=False)
# plt.show()


train_1 = pd.DataFrame(df, index = train.index)
test_1 = pd.DataFrame(df, index = test.index)

actives = train_1.loc[(train_1["forecast_3_month"]>0)&(train_1["sales_9_month"]>0)]
# from collections import Counter
# print('Original dataset shape {}'.format(Counter(train_1.went_on_backorder)))
# print('Reduced dataset shape {}'.format(Counter(actives.went_on_backorder)))
# print('Class 0 reduction: %.2f%%' % (100*(1 - Counter(actives.went_on_backorder)[0]/Counter(train_1.went_on_backorder)[0])))
# print('Class 1 reduction: %.2f%%' % (100*(1 - Counter(actives.went_on_backorder)[1]/Counter(train_1.went_on_backorder)[1])))

# n_components = 2

# pca1 = PCA(2).fit(actives.iloc[:,:-1])
# X1 = pca1.transform(actives.iloc[:,:-1])
# plt.figure(figsize=(10,8))
# plt.scatter(X1[:, 0], X1[:, 1], c=actives.went_on_backorder, s=10, cmap=plt.cm.jet)
# plt.show()


X2 = PCA(3).fit_transform(actives.iloc[:,:-1])


def plot_pca(azim):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d');
    ax.scatter(X2[:,0], X2[:,1], X2[:,2], c=actives.went_on_backorder, s=3, cmap=plt.cm.jet, alpha=1)
    ax.view_init(20, azim)

plot_pca(-60)
plot_pca(-20)
plt.show()