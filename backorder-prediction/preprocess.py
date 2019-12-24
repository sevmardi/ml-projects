import pandas as pd
import numpy as np
import orderclassifier
import random


def read_data(file_path):
    print()
    backorders = pd.read_csv(file_path)
    np_orders = backorders.as_matrix()
    data = convert_instance(np_orders)
    print(data)

    return data


def convert_boolean(value):
	"""
	given string value, convert to boolean
	"""
	if value == 'No':
	    return 0
	elif value == 'Yes':
	    return 1

	else:
	    return -99


def convert_instance(x):
	"""
	convert x instsance into boolean
	"""
	print()
	for row in x:
	    if np.isnan(row[2]):
	        row[2] = -99
	    row[12] = convert_boolean(row[12])
	    for i in range(17, 23):
	        row[i] = convert_boolean(row[i])

	return x


def divide_dataset(data, train_prob):
    """
    Split dataset
    """
    print()
    train = []
    valid = []
    for i in range(0, data.shape[0]):
        prob = random.uniform(0, 1)
        if prob < train_prob:
            train.append(data[i])
        else:
            valid.append(data[i])

    train = np.array(train)
    valid = np.array(valid)

    return train, valid


def write_datafile(data, file_path):
    """
    Write data file
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, header=None, index=False)


def write_dataset(train, valid, train_file, valid_file):
    """
        Write dataset
    """
    write_datafile(train, train_file)
    write_datafile(valid, valid_file)


def pre_process():
    data = read_data("data/train.csv")
    print(data.shape)
    train, valid = divide_dataset(data, 0.8)
    print(train.shape)
    print(valid.shape)
    write_dataset(train, valid, "data/d_train_set.csv", "data/d_valid_set.csv")

if __name__ == '__main__':
    pre_process()
    data = read_data('data/train.csv')
    # orderclassifier.count_pos_neg(data)
