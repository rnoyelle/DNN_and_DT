from lib.network import create_network
from lib.datasets import DataManager

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import csv

# Network params
dim_embedding = 50
dim_dense = 100

# training params
batch_size = 128
n_epoch = 2  # 10  # 500
patience = 3  # 50  # Early Stopping

# Loading the data
DM = DataManager(dataset_path='../datasets')
supported_dataset = DM.get_supported_dataset()
print('Dataset available :', supported_dataset)

for data_name in ['Covertype']:  #supported_dataset: # 'Covertype'
    dirName = '../results/' + re.sub('[^a-zA-Z0-9]+', '', data_name)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    print(data_name)

    X, y, features_names, class_names = DM.get_data(data_name)

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

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    # Create the Dense Neural Network
    model, clf = create_network(X, np.argmax(y, axis=1),
                                dim_embedding=dim_embedding, dim_dense=dim_dense,
                                return_tree=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # model callbacks
    early_stop = EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience / 2), verbose=1)
    log_file_path = dirName + '/training_file_{}.log'.format(data_name)
    csv_logger = CSVLogger(log_file_path, append=False)
    # model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
    # model_checkpoint = ModelCheckpoint(model_names,
    #                                    monitor='val_loss',
    #                                    verbose=1,
    #                                    save_best_only=True,
    #                                    save_weights_only=False)
    callbacks = [csv_logger, early_stop, reduce_lr]  # [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # Fitting the model
    history = model.fit([X_train[:, i] for i in range(X_train.shape[1])], y_train,
                        epochs=n_epoch, batch_size=batch_size, shuffle=True, callbacks=callbacks,
                        validation_data=([X_test[:, i] for i in range(X_test.shape[1])], y_test))

    loss_train, acc_train = model.evaluate([X_train[:, i] for i in range(X_train.shape[1])], y_train)
    loss_test, acc_test = model.evaluate([X_test[:, i] for i in range(X_test.shape[1])], y_test)

    # serialize model to JSON
    model_json = model.to_json()
    with open(dirName + '/model_{}.json'.format(data_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(dirName + '/model_{}.h5'.format(data_name))
    # print("Saved model to disk")

    filename = '../results/perf_tree_DNN.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as csvfile:
        headers = ['data_name', 'loss_train', 'acc_train', 'loss_test', 'acc_test']
        writer = csv.DictWriter(csvfile, delimiter='\t', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow({'data_name': data_name,
                         'loss_train': loss_train, 'acc_train': acc_train,
                         'loss_test': loss_test, 'acc_test': acc_test})

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(dirName + '/history_training_accuracy.png', bbox_inches='tight')
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(dirName + '/history_training_loss.png', bbox_inches='tight')
    plt.clf()

    # Plot the DNN model
    from keras.utils import plot_model

    plot_model(model, to_file=dirName + '/model_architecture_{}.pdf'.format(data_name))  #png

    # plot the Tree (need feature name & class name !)
    from sklearn import tree
    import graphviz

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=features_names,
                                    class_names=class_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    # tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(dirName + '/decision_tree_{}.pdf'.format(data_name))
    # graph
