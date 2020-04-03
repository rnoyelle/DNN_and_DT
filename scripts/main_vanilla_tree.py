# from lib.network import create_network
from lib.datasets import DataManager

from sklearn.tree import DecisionTreeClassifier

# from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
# from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import csv

# Loading the data
DM = DataManager(dataset_path='../datasets')
supported_dataset = DM.get_supported_dataset()
print('Dataset available :', supported_dataset)

for data_name in supported_dataset:
    # dirName = '../results/' + re.sub('[^a-zA-Z0-9]+', '', data_name)
    #     # if not os.path.exists(dirName):
    #     #     os.mkdir(dirName)
    print(data_name)

    X, y, features_names, class_names = DM.get_data(data_name)

    n_feature = X.shape[1]
    classes = len(np.unique(y))
    # print(classes)
    print(X.shape, y.shape)

    # preprocessing the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # Create the Decision Tree
    clf = DecisionTreeClassifier(random_state=0, max_depth=10)
    clf.fit(X_train, y_train)
    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)

    filename = '../results/perf_vanilla_tree.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as csvfile:
        headers = ['data_name', 'acc_train', 'acc_test']
        writer = csv.DictWriter(csvfile, delimiter='\t', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow({'data_name': data_name, 'acc_train': acc_train, 'acc_test': acc_test})
