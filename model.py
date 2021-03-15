import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import L1, L2
import numpy as np

def Ksquare(x):
    return tf.pow(x, 2)

def Klog(x):
    return tf.math.log(x)

class Model:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    # TA's example
    def rnn1(self):

        model = models.Sequential()
        model.add(layers.Permute((2, 1), input_shape=self.input_dim)) # (batch, 22, 1000) -> (batch, 1000, 22)
        model.add(layers.SimpleRNN(self.hidden_dim)) # (batch, 1000, 22) -> (batch, hidden_dim)
                                                # Use return_sequences=True to use in cascaded RNNs
        model.add(layers.Dense(4, activation='softmax')) # (batch, hidden_dim) -> (batch, 4
            # Need to compile model before running
            # use optimizer='sgd' if desired. Seems to be case-insensitive
            # use loss = 'categorical_crossentropy' for one-hot labels
            # see https://www.tensorflow.org/api_docs/python/tf/keras/metrics for other metrics.
            #     Metrics might not have string names.

        return model

    # 73.8% accuracy (max: 76%)
    def ResNet(self):

        inputs = layers.Input(shape=self.input_dim)

        r1 = layers.Reshape(self.input_dim + (1,))(inputs)
        p1 = layers.Permute((2, 3, 1))(r1)
        c1 = layers.Conv2D(200, (10, 1), activation='elu', padding='same')(p1)
        mp1 = layers.MaxPool2D((3, 1), padding='same')(c1)
        b1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(mp1)
        dp = layers.Dropout(0.5)(b1)

        for _ in range(3):

            c2 = layers.Conv2D(200, (15, 1), padding='same')(dp)
            b2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c2)

            c3 = layers.Conv2D(200, (15, 1), activation='elu', padding='same')(b2)
            b3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c3)

            con = layers.add([dp, b3])

            mp = layers.MaxPool2D((3, 1), padding='same')(con)
            dp = layers.Dropout(0.5)(mp)

        r1 = layers.Reshape((4, -1))(dp)
        f1 = layers.Flatten()(r1)

        outputs = layers.Dense(4, activation='softmax', kernel_regularizer=L1(0.01), activity_regularizer=L2(0.01))(f1)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ResNet')

        return model

if __name__ == "__main__":
    model = Model((22, 250), 20)
    rnn = model.cnn2()
    rnn.summary()
