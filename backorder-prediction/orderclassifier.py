import numpy as np
import pandas as pd
import preprocess as o_pre
import random
import time

from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPClassifier


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_dataset(train_prob):
    file_path = "/data/train.csv"
    backorders = pd.read_csv(file_path)
    print(backorders)

    npOrders = backorders.as_matrix()

    np.random.shuffle(npOrders)

    total = npOrders.shape[0]
    train_size = int(total * train_prob)
    train, valid = npOrders[:train_size, :], npOrders[train_size:, :]

    return (train, valid)


def train_svm_model(train):
    # raise NotImplementedError('Not implemented')
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(train[:, 1:-1], train[:, -1])
    return clf


def train_mlp_model(train):
    # raise NotImplementedError('Not implemented')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(size,), random_state=1)
    clf.fit(train[:, 1:-1], train[:, -1])
    return clf


def train_rf_model(train):
    # raise NotImplementedError('Not implemented')
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(train[:, -1:-1], train[:, -1])
    return clf


def valid_model(valid, clf):
    # raise NotImplementedError('Not implemented')
    result = clf.fit(valid[:, 1:-1])
    return result


def count_pos_neg(data):
    pos = 0
    neg = 0
    for row in data:
        if(row[-1] == 0):
            neg += 1
        else:
            pos += 1
    print(pos, neg)

    return (pos, neg)


def count_results(data, result):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(0, result.shape[0]):
        if data[i, -1] == 1 and result[i] == 1:
            true_pos += 1
        elif data[i, -1] == 1 and result[i] == 0:
            false_neg += 1
        elif data[i, -1] == 0 and result[i] == 1:
            false_pos += 1
        elif data[i, -1] == 0 and result[i] == 0:
            true_neg += 1
    print(true_pos, true_neg, false_pos, false_neg)

    return (true_pos, true_neg, false_pos, false_neg)


def eval_result(true_pos, true_neg, false_pos, false_neg):
    # raise NotImplementedError('Not implemented')
    precision = true_pos * 1.0 / (true_pos + false_pos)
    recall = true_pos * 1.0 / (true_pos + false_neg)
    f_score = (precision * recall) * 2.0 / (precision + recall)
    print(precision, recall, f_score)
    return (precision, recall)


def eval_model(clf, data):
    # raise NotImplementedError('Not implemented')
    result = clf.predict(data[:, 1:22])
    print(result.shape)
    print(result)
    true_pos, true_neg, false_pos, false_neg = count_results(data, result)
    eval_result(true_pos, true_neg, false_pos, false_neg)


def predict(data, clf):
    # raise NotImplementedError('Not implemented')
    result = clf.predict(data[:, 1:22])
    return result


def write_model(clf, file_path):
    # raise NotImplementedError('Not implemented')
    joblib.dump(clf, file_path)


def read_model(file_path):
    # raise NotImplementedError('Not implemented')
    clf = joblib.load(file_path)
    return clf


def write_numpy(data, file_path):
    # raise NotImplementedError('Not implemented')
    df = pd.DataFrame(data=data)
    df.to_csv(file_path, sep=',', header=False, index=False)


def write_data(train, valid, train_path, valid_path):
    # raise NotImplementedError('Not implemented')
    write_numpy(train, train_path)
    write_numpy(valid, valid_path)


def train_write(train, file_path):
    # raise NotImplementedError('Not implemented')
    start_time = time.time()
    clf = train_rf_model(train)
    total_time = time.time() - start_time
    print(total_time)

    write_model(clf, file_path)


def writeData():
    # raise NotImplementedError('Not implemented')
    (train, valid) = read_dataset(0.1)
    print(train.shape)
    count_pos_neg(train)

    print(valid.shape)
    count_pos_neg(valid)

    write_data(train, valid, 'backorder_train.txt', 'backorder_valid.txt')


def read_numpy(file_path):
    # raise NotImplementedError('Not implemented')
    data = pd.read_csv(file_path)
    data_np = data.as_matrix()
    return data_np


def read_data(train_path, valid_path):
    # raise NotImplementedError('Not implemented')
    train = read_numpy(train_path)
    valid = read_numpy(valid_path)
    return (train, valid)


def sample_data(data, neg_prob):
    # raise NotImplementedError('Not implemented')
    result = []
    for i in range(0, data.shape[0]):
        if data[i, -1] == 0:
            prob = random.uniform(0, 1)
            if prob < neg_prob:
                result.append(data[i])
        else:
            result.append(data[i])

    return np.array(result)


def duplicate_data(data, pos_times):
    # raise NotImplementedError('Not implemented')
    result = []
    for i in range(0, data.shape[0]):
        if data[i, -1] == 1:
            for t in range(0, pos_times):
                result.append(data[i])
        else:
            result.append(data[i])
    return np.array(result)


def plot_roc(valid, clf, model_name):
    # raise NotImplementedError('Not implemented')
    probs = clf.predict_proba(valid[:, 1:22])
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(valid[:, -1], preds)
    roc_auc = metrics.auc(fpr, tpr)

    # roc
    plt.title('AUC' + model_name)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive rate')
    plt.xlabel('False Positive rate')
    plt.show()

    # precision-racall-curve
    precision, recall, thresholds = precision_recall_curve(valid[:, -1], preds)
    area = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label='Area Under Curve = %0.3f' % area)
    plt.legend(loc='lower left')
    plt.title('Precision-Recall ' + model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.show()


def train_from_csv_data(data_file, model_file):
    # raise NotImplementedError('Not implemented')
    data = o_pre.read_data(data_file)
    train_write(data, model_file)


def build_classifier_sample():
    # raise NotImplementedError('Not implemented')
    train, valid = read_data('data/train.csv', 'data/test.csv')
    print(valid)

    print(valid[:, 1:-1].shape)
    count_pos_neg(valid)
    print(train[:, 1:-1].shape)
    count_pos_neg(train)

    duplicate_train = duplicate_data(train, 10)
    count_pos_neg(duplicate_data)

    train_write(duplicate_data, 'model/rf_d_duplicate_10.model')


def eval_result_sample(model_file, model_name):
    # raise NotImplementedError('Not implemented')
    train, valid = read_data('data/d_train_set.csv', 'data/d_valid_set.csv')
    print(valid[:, 1:-1].shape)
    count_pos_neg(valid)
    print(valid[:, 1:-1].shape)
    count_pos_neg(train)

    clf = read_model(model_file)
    eval_model(clf, valid)
    plot_roc(valid, clf, model_name)


def eval_result_rf():
    # raise NotImplementedError('Not implemented')
    eval_result_sample("model/rf_d_duplicate.model", "random forest")


if __name__ == '__main__':
    print()
    eval_result_rf()
