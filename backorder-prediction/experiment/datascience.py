import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('data/train.csv')

quantfeature = ['national_inv',
                'lead_time',
                'in_transit_qty',
                'forecast_3_month',
                'forecast_6_month',
                'forecast_9_month',
                'sales_1_month',
                'sales_3_month',
                'sales_6_month',
                'sales_9_month',
                'min_bank',
                'pieces_past_due',
                'perf_6_month_avg',
                'perf_12_month_avg',
                'local_bo_qty']
binfeature = ['potential_issue',
              'deck_risk',
              'oe_constraint',
              'ppap_risk',
              'stop_auto_buy',
              'rev_stop',
              'went_on_backorder']

pred_binfeature = ['potential_issue',
                   'deck_risk',
                   'oe_constraint',
                   'ppap_risk',
                   'stop_auto_buy',
                   'rev_stop']

for col in binfeature:
    data[col] = pd.factorize(data[col])[0]

data['perf_6_month_avg'] = data['perf_6_month_avg'].replace(-99, np.NaN)
data['perf_12_month_avg'] = data['perf_12_month_avg'].replace(-99, np.NaN)
data = data.fillna(data.median(), inplace=True)

y_train = data['went_on_backorder']
X_train = data.drop(['sku', 'went_on_backorder'], axis=1)

big_train = data[y_train == 0]
small_train = data[y_train == 1]
len_small_train = len(small_train)

big_train_lower_sampled = resample(big_train,
                                   replace=False,
                                   n_samples=len_small_train,
                                   random_state=123)
data_lower_sampled = pd.concat([big_train_lower_sampled, small_train])
y_train_new = data_lower_sampled['went_on_backorder']
X_train_new = data_lower_sampled.drop(['sku', 'went_on_backorder'], axis=1)


data_test = 'data/test.csv'

for col in pred_binfeature:
    data_test[col] = pd.factorize(data_test[col])[0]
X_test = data_test.drop(['sku'], axis=1)
rf = RandomForestClassifier(n_estimators=50, max_features=3)
rf.fit(X_train_new, y_train_new)
y_test = rf.predict(X_test)

print(y_test)
