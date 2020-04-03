# import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree

# import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model

# import os
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from keras.utils import to_categorical
#
#
# import matplotlib.pyplot as plt


def number_of_slaves(children_left, children_right, feature, node_id):
    number_of_slaves = 2

    if feature[children_left[node_id]] < 0:
        number_of_slaves -= 1
    if feature[children_right[node_id]] < 0:
        number_of_slaves -= 1

    return(number_of_slaves, feature[children_left[node_id]] < 0, feature[children_right[node_id]] < 0)


def get_output(children_left, children_right, feature, X_emb, dim_dense, node_id=0):
    s, left_is_leave, right_is_leave = number_of_slaves(children_left, children_right, feature, node_id)

    if left_is_leave and right_is_leave:
        return X_emb[feature[node_id]]

    elif left_is_leave and not right_is_leave:
        h = get_output(children_left, children_right, feature, X_emb, dim_dense,
                       node_id=children_right[node_id])
        return Dense(dim_dense, activation='relu')(concatenate([X_emb[feature[node_id]], h]))

    elif right_is_leave and not left_is_leave:
        h = get_output(children_left, children_right, feature, X_emb, dim_dense,
                       node_id=children_left[node_id])
        return Dense(dim_dense, activation='relu')(concatenate([h, X_emb[feature[children_right[node_id]]]]))

    else:
        h_left = get_output(children_left, children_right, feature, X_emb, dim_dense,
                            node_id=children_left[node_id])
        h_right = get_output(children_left, children_right, feature, X_emb, dim_dense,
                             node_id=children_right[node_id])
        h = Dense(dim_dense, activation='relu')(concatenate([h_left, h_right]))
        return Dense(dim_dense, activation='relu')(concatenate([h, X_emb[feature[node_id]]]))


def create_network(X, y,
                   dim_embedding=50, dim_dense=100, return_tree=False):
    """
    :param X: numpy array of shape [n_samples, n_features]
    :param y: numpy array of shape [n_samples]
    :param dim_embedding: int, size of the embedding feature
    :param dim_dense: int, size of the inner dense layers
    :param return_tree: bool, if set to true, Decision Tree clf will be returned

    :return: keras model (uncompiled)

    """
    n_feature = X.shape[1]
    classes = len(np.unique(y))

    # Create Tree
    clf = DecisionTreeClassifier(random_state=0, max_depth=10)
    clf.fit(X, y)

    # plt.figure(figsize=(15, 15))
    # tree.plot_tree(clf)
    # plt.show()

    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature

    # Create Dense Neural Network
    inputs = [Input(shape=(1,)) for _ in range(n_feature)]
    X_emb = [Dense(dim_embedding, activation='relu')(el) for el in inputs]

    h = get_output(children_left, children_right, feature, X_emb, dim_dense, node_id=0)
    h = Dense(classes, activation='softmax')(h)

    model = Model(inputs, h)

    if return_tree:
        return model, clf
    else:
        return model

def create_network_from_df(df, numerical_features, categorical_features, y,
                   dim_embedding=50, dim_dense=100, return_tree=False, max_depth=None):

    classes = len(np.unique(y))

    # Create Tree
    clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
    clf.fit(df.values, y)

    # plt.figure(figsize=(15, 15))
    # tree.plot_tree(clf)
    # plt.show()

    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature

    # Create Dense Neural Network
    inputs = [Input(shape=(1,)) for col_name in df.columns]
    # Create Embedding layer like : Dense layer for numerical feature, Embedding layer for categorical feature
    X_emb = []
    for i, col_name in enumerate(df.columns) :
        if col_name in numerical_features :
            X_emb.append(Dense(dim_embedding, activation=None)(inputs[i]))
        else : # categorical feature
            cat_size = df[col_name].nunique()
            X_emb.append(Flatten()(Embedding(cat_size, dim_embedding, input_length=1)(inputs[i])))

    h = get_output(children_left, children_right, feature, X_emb, dim_dense, node_id=0)
    h = Dense(classes, activation='softmax')(h)

    model = Model(inputs, h)

    if return_tree:
        return model, clf
    else:
        return model



