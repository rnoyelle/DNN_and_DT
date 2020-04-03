from lib.network import create_network
from lib.datasets import DataManager

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

# Network params
dim_embedding = 50
dim_dense = 100

batch_size = 20
n_epoch = 500

# Loading the data
# dataset_name = 'iris'
# DM = DataManager(dataset_name=dataset_name)
# X, y = DM.get_data()

DM = DataManager(dataset_path='../datasets')
X, y, _, _ = DM.get_data('iris')



n_feature = X.shape[1]
classes = len(np.unique(y))
# print(classes)
print(X.shape, y.shape)

# preprocessing the data
y = to_categorical(y, num_classes=classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(X_train.shape)
# print(X_test.shape)

# Create the Dense Neural Network
model, clf = create_network(X, np.argmax(y, axis=1),
                       dim_embedding=dim_embedding, dim_dense=dim_dense,
                       return_tree=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the model
history = model.fit([X_train[:, i] for i in range(X_train.shape[1])], y_train,
          epochs=n_epoch, batch_size=batch_size, shuffle=True,
          validation_data=([X_test[:, i] for i in range(X_test.shape[1])], y_test))


# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# Plot the DNN model
# from keras.utils import plot_model
# plot_model(model, to_file='model_{].png'.format(dataset_name))

# plot the Tree (need feature name & class name !)
# from sklearn import tree
# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None,
#                      feature_names=iris.feature_names,
#                      class_names=iris.target_names,
#                      filled=True, rounded=True,
#                      special_characters=True)
# tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris_tree")
# graph
















