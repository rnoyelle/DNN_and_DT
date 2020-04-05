import os
import pandas as pd
import numpy as np

from lib.network import create_network_from_df
# from lib.datasets import DataManager

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau


# Network params
dim_embedding = 50
dim_dense = 100

# training params
batch_size = 128
n_epoch = 500  # 500
patience = 50   # Early Stopping
max_depth = 5


"""
Load data as DataFrame
"""
data_path = os.path.join('../datasets/', 'German_Credit_Data/german.data')

labels = ['Status of existing checking account', 'Duration in month', 'Credit history', 'Purpose', 'Credit amount',
          'Savings account/bonds', 'Present employment since', 'Installment rate in percentage of disposable income',
          'Personal status and sex', 'Other debtors / guarantors', 'Present residence since', 'Property',
          'Age in years',
          'Other installment plans ', 'Housing', 'Number of existing credits at this bank', 'Job',
          'Number of people being liable to provide maintenance for', 'Telephone', 'foreign worker',
          'Class']

df = pd.read_csv(data_path, sep='\s+', header=None, names=labels)

class_col = 'Class'
df['Class'] = df['Class'] - 1
classes = df[class_col].nunique()
y = df.pop(class_col)


"""
Encode categorical data into int 
"""
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
categorical_features = [col for col in df.columns if col not in numerical_features]

for col in categorical_features :
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.codes


# preprocessing the data
y = to_categorical(y, num_classes=classes)
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
df_train = df_train.copy()
df_test = df_test.copy()
if len(numerical_features)>0 :
    scaler = StandardScaler()
    scaler.fit(df_train[numerical_features])
    df_train.loc[:, numerical_features] = scaler.transform(df_train[numerical_features])
    df_test.loc[:, numerical_features] = scaler.transform(df_test[numerical_features])
"""
Create network
"""

model, clf = create_network_from_df(df_train, numerical_features, categorical_features, np.argmax(y_train, axis=1),
                                    dim_embedding=dim_embedding, dim_dense=dim_dense,  max_depth=max_depth, return_tree=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""
training the network
"""
# model callbacks
early_stop = EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience / 2), verbose=1)

callbacks = [early_stop, reduce_lr]  # [model_checkpoint, csv_logger, early_stop, reduce_lr]


history = model.fit([df_train[col] for col in df_train.columns], y_train,
                    epochs=n_epoch, batch_size=batch_size, shuffle=True,
                    callbacks=callbacks,
                   validation_data=([df_test[col] for col in df_test.columns], y_test))



