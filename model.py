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
    
    def rnn2(self):

        # Equivalent functional declaration:
        inputs = layers.Input(shape=(22, 1000), name='inputs')
            # Needed for first layer. Expects input of (batch, 22, 1000)
            # We can also name our layers. Useful for getting a layer by a string name.
        p1 = layers.Permute((2, 1), name='p1')(inputs)
            # Format is ClassName(*args, **kwargs)(upstream_layer)
            # Permute: count batch dimension as 0
        rnn = layers.SimpleRNN(self.hidden_dim, name='rnn')(p1)
        outputs = layers.Dense(4, activation='softmax', name='outputs')(rnn)
        model = models.Model(inputs=inputs, outputs=outputs, name='functional_model') # or keras.Model(*, **)
            # Need to declare a model specifying input and output layers.
            # Can pass lists of layers instead.
            # We can also name our model.

        return model

    def ShallowConvNet(self):

        inputs = layers.Input(shape=(22, 1000))
        r1 = layers.Reshape((22, 1000, 1))(inputs)
            # (N, 22, 1000) -> (N, 22, 1000, 1)
        c1 = layers.Conv2D(40, (1, 25), activation='elu', data_format='channels_last')(r1)
            # (N, 22, 1000, 1) -> (N, 22, 976, 40), i.e. NHWC -> NHWC. 'channels_last' is default
        p1 = layers.Permute((2, 1, 3))(c1)
            # (N, 22, 976, 40) -> (N, 976, 22, 40)
        r2 = layers.Reshape((976, 22*40))(p1)
            # (N, 976, 22, 40) -> (N, 976, 22*40)
        d1 = layers.Dense(40, activation='elu')(r2)
            # (N, 976, 22*40) -> (N, 976, 40)
            # weight_shape = 22*40 x 40 = 35200
            # bias_shape = 40
        sq1 = layers.Activation(Ksquare)(d1)
        ap1 = layers.AveragePooling1D(75, strides=15)(sq1)
            # (N, 976, 40) -> (N, 61, 40)
        log1 = layers.Activation(Klog)(ap1)
        f1 = layers.Flatten()(log1)
            # (N, 61, 40) -> (N, 61*40)
        outputs = layers.Dense(4, activation='softmax')(f1)
            # (N, 61*40) -> (N, 4)

        model = models.Model(inputs=inputs, outputs=outputs, name='shallow_convnet')
        
        #model.compile(keras.optimizers.Adam(), keras.losses.SparseCategoricalCrossentropy(), 
        #             metrics=[keras.metrics.SparseCategoricalAccuracy()])

        return model

    def rnn_LSTM(self):

        inputs = layers.Input(shape=self.input_dim)

        r1 = layers.Reshape((22, 1000, 1))(inputs)
            # (N, 22, 1000) -> (N, 22, 1000, 1)
        c1 = layers.Conv2D(40, (1, 25), activation='elu', data_format='channels_last')(r1)
            # (N, 22, 1000, 1) -> (N, 22, 976, 40), i.e. NHWC -> NHWC. 'channels_last' is default
        p1 = layers.Permute((2, 1, 3))(c1)
            # (N, 22, 976, 40) -> (N, 976, 22, 40)
        r2 = layers.Reshape((976, 22*40))(p1)
            # (N, 976, 22, 40) -> (N, 976, 22*40)
        d1 = layers.Dense(40, activation='elu')(r2)
            # (N, 976, 22*40) -> (N, 976, 40)
            # weight_shape = 22*40 x 40 = 35200
            # bias_shape = 40
        sq1 = layers.Activation(Ksquare)(d1)
        ap1 = layers.AveragePooling1D(75, strides=15)(sq1)
            # (N, 976, 40) -> (N, 61, 40)
        log1 = layers.Activation(Klog)(ap1)
        f1 = layers.Flatten()(log1)
            # (N, 61, 40) -> (N, 61*40) 
        
        r3 = layers.Reshape((61*40, 1))(f1)
        
        lstm1 = layers.LSTM(10)(r3)
        dp1 = layers.Dropout(0.2)(lstm1)

        outputs = layers.Dense(4, activation='softmax', kernel_regularizer=L1(0.01), activity_regularizer=L2(0.01))(dp1)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='LSTM')

        return model

if __name__ == "__main__":
    model = Model((22, 1000), 20)
    rnn = model.rnn1()
    rnn.summary()
