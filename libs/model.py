import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import TimeDistributed, Conv2D


def get_model():

    model = Sequential()

    # Encoder
    model.add(ConvLSTM2D(
        filters=20, kernel_size=(3, 3),
        padding='same', return_sequences=False,
        activation='tanh', recurrent_activation='sigmoid',
        input_shape=(48, 74, 99, 1)))

    def repeat_last_status(x):
        x = tf.reshape(x, (-1, 1, 74, 99, 20))
        copied_x = tf.identity(x)
        for _ in range(47):
            x = tf.concat([x, copied_x], axis=1)
        return x
    model.add(Lambda(repeat_last_status))

    # Decoder
    model.add(ConvLSTM2D(
        filters=20, kernel_size=(3, 3),
        padding='same', return_sequences=True,
        activation='tanh', recurrent_activation='sigmoid'))

    model.add(TimeDistributed(
        Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')))

    model.compile(loss="mae", optimizer="adam")

    return model
