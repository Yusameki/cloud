import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from libs.model import get_model

# Get train and val data
train_sat_dataset = np.load('./datasets/train_sat.npy')
train_sat = train_sat_dataset[:365]
val_sat = train_sat_dataset[365:]
train_sat = train_sat.astype(np.float32) / 255
val_sat = val_sat.astype(np.float32) / 255

X_index = np.arange(364)
Y_index = np.arange(1, 365)

X_train = train_sat[X_index]
Y_train = train_sat[Y_index]

X_indexv = np.arange(306)
Y_indexv = np.arange(1, 307)

X_val = val_sat[X_indexv]
Y_val = val_sat[Y_indexv]

# set model
model = get_model()

validation_data = (X_val, Y_val)

# EarlyStopping
e_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Checkpoint
checkpoint = ModelCheckpoint(filepath='./models/my_model.h5',
                             monitor='val_loss', save_best_only=True, save_weights_only=True)

callbacks = [e_stopping, checkpoint]

# training
model.fit(x=X_train, y=Y_train, batch_size=8, epochs=100,
          validation_data=validation_data, callbacks=callbacks)
