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
    
    # TA's example
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

    # TA's example
    # 62% accuracy (max: 66%)
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

    # 65 % accuracy (max: 68%)
    def cnn1(self):

        inputs = layers.Input(shape=self.input_dim)

        # [conv -> elu -> bn] x3 -> avg_pool -> dp -> fc
        r1 = layers.Reshape((22, 1000, 1))(inputs)
            # (N, 22, 1000) -> (N, 22, 1000, 1)
        c1 = layers.Conv2D(25, (1, 10), activation='elu', data_format='channels_last')(r1)
            # (N, 22, 1000, 1) -> (N, 22, 991, 25), i.e. NHWC -> NHWC. 'channels_last' is default
        b1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c1)
            # (N, 22, 991, 25) -> (N, 22, 991, 25)
        c2 = layers.Conv2D(25, (3, 3), activation='elu', data_format='channels_last')(b1)
            # (N, 22, 991, 25) -> (N, 20, 989, 25)
        b2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c2)
            # (N, 20, 989, 25) -> (N, 20, 989, 25)
        c3 = layers.Conv2D(25, (18, 1), activation='elu', data_format='channels_last')(b2)
            # (N, 20, 989, 25) -> (N, 3, 989, 25)
        b3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c3)
            # (N, 3, 989, 25) -> (N, 3, 989, 25)
        p1 = layers.Permute((2, 1, 3))(b3)
            # (N, 3, 989, 25) -> (N, 989, 3, 25)
        r1 = layers.Reshape((989, 3*25))(p1)
            # (N, 989, 3, 25) -> (N, 989, 3*25)
        mp1 = layers.AveragePooling1D(74, strides=15)(r1)
            # (N, 989, 75) -> (N, 62, 75)
        dp1 = layers.Dropout(0.2)(mp1)
        f1 = layers.Flatten()(dp1)
            # (N, 62, 75) -> (N, 62*75)

        outputs = layers.Dense(4, activation='softmax', kernel_regularizer=L1(0.01), activity_regularizer=L2(0.01))(f1)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='cnn1')

        return model

    # 68% accuracy (max: 72%)
    def cnn2(self):

        inputs = layers.Input(shape=self.input_dim)

        # [conv -> elu -> conv -> elu -> bn -> dp] x3 -> avg_pool -> fc
        r1 = layers.Reshape((22, 1000, 1))(inputs)
            # (N, 22, 1000) -> (N, 22, 1000, 1)
        c1 = layers.Conv2D(20, (1, 10), activation='elu', data_format='channels_last')(r1)
            # (N, 22, 1000, 1) -> (N, 22, 991, 40), i.e. NHWC -> NHWC. 'channels_last' is default
        c11 = layers.Conv2D(20, (3, 3), activation='elu', data_format='channels_last')(c1)
            # (N, 22, 991, 20) -> (N, 20, 989, 20), i.e. NHWC -> NHWC. 'channels_last' is default
        b1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c11)
            # (N, 20, 989, 20) -> (N, 20, 989, 20)
        dp1 = layers.Dropout(0.1)(b1)

        c2 = layers.Conv2D(20, (3, 3), activation='elu', data_format='channels_last')(dp1)
            # (N, 20, 989, 20) -> (N, 18, 987, 20)
        c22 = layers.Conv2D(20, (3, 3), activation='elu', data_format='channels_last')(c2)
            # (N, 18, 987, 20) -> (N, 163, 985, 20)
        b2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c22)
            # (N, 16, 985, 20) -> (N, 16, 985, 20)
        dp2 = layers.Dropout(0.1)(b2)

        c3 = layers.Conv2D(20, (3, 3), activation='elu', data_format='channels_last')(dp2)
            # (N, 16, 985, 20) -> (N, 14, 983, 20)
        c33 = layers.Conv2D(20, (12, 1), activation='elu', data_format='channels_last')(c3)
            # (N, 14, 983, 20) -> (N, 3, 983, 20)
        b3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c33)
            # (N, 3, 983, 20) -> (N, 3, 983, 20)
        dp3 = layers.Dropout(0.1)(b3)

        p1 = layers.Permute((2, 1, 3))(dp3)
            # (N, 3, 983, 20) -> (N, 983, 3, 20)
        r1 = layers.Reshape((983, 3*20))(p1)
            # (N, 983, 3, 20) -> (N, 983, 3*20)
        mp1 = layers.AveragePooling1D(78, strides=15)(r1)
            # (N, 983, 60) -> (N, 61, 60)

        f1 = layers.Flatten()(mp1)
            # (N, 61, 60) -> (N, 61*60)

        outputs = layers.Dense(4, activation='softmax', kernel_regularizer=L1(0.01), activity_regularizer=L2(0.01))(f1)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='cnn3')

        return model

    # 67% accuracy (max: 69%)
    def cnn3(self):

        inputs = layers.Input(shape=self.input_dim)

        # [conv -> conv -> bn -> elu] x2 -> avg_pool -> fc
        r1 = layers.Reshape((22, 1000, 1))(inputs)
            # (N, 22, 1000) -> (N, 22, 1000, 1)
        c1 = layers.Conv2D(20, (1, 10), data_format='channels_last')(r1)
            # (N, 22, 1000, 1) -> (N, 22, 991, 20)
        c11 = layers.Conv2D(20, (3, 3), data_format='channels_last')(c1)
            # (N, 20, 989, 20) -> (N, 22, 989, 20)
        b1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c11)
        elu1 = layers.Activation('elu')(b1)
            # (N, 20, 989, 20) -> (N, 20, 989, 20)

        c2 = layers.Conv2D(20, (3, 3), data_format='channels_last')(elu1)
            # (N, 20, 989, 20) -> (N, 18, 987, 20)
        c22 = layers.Conv2D(20, (3, 3), data_format='channels_last')(c2)
            # (N, 18, 987, 20) -> (N, 3, 985, 20)
        b2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c22)
        elu2 = layers.Activation('elu')(b2)
            # (N, 3, 985, 20) -> (N, 3, 985, 20)

        c3 = layers.Conv2D(20, (3, 3), data_format='channels_last')(elu2)
            # (N, 3, 985, 20) -> (N, 3, 982, 20)
        c33 = layers.Conv2D(20, (12, 1), data_format='channels_last')(c3)
            # (N, 3, 982, 20) -> (N, 3, 974, 20)
        b3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c33)
        elu3 = layers.Activation('elu')(b3)
            # (N, 3, 974, 20) -> (N, 3, 974, 20)

        p1 = layers.Permute((2, 1, 3))(elu3)
            # (N, 3, 974, 20) -> (N, 974, 3, 20)
        r1 = layers.Reshape((983, 3*20))(p1)
            # (N, 974, 3, 20) -> (N, 974, 3*20)
        mp1 = layers.AveragePooling1D(78, strides=15)(r1)
            # (N, 974, 60) -> (N, 181, 60)

        # lstm = layers.GRU(50, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(mp1)
            # (N, 91, 60) -> (N, 91, 50)

        f1 = layers.Flatten()(mp1)
            # (N, 91, 50) -> (N, 91*50)

        outputs = layers.Dense(4, activation='softmax', kernel_regularizer=L1(0.01), activity_regularizer=L2(0.01))(f1)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='cnn3')

        return model

    # 64% accuracy (max: 66%)
    def ResNet(self):

        inputs = layers.Input(shape=self.input_dim)

        # conv -> elu-> [ResNet block: conv -> bn -> elu -> conv -> bn -> concat -> elu] -> conv -> avg_pool -> fc
        r1 = layers.Reshape((22, 1000, 1))(inputs)
        c1 = layers.Conv2D(20, (1, 10), data_format='channels_last')(r1)
        elu = layers.Activation('elu')(c1)

        for _ in range(6):
            c2 = layers.Conv2D(20, (3, 3), data_format='channels_last', padding='same')(elu)
            b1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c2)
            elu1 = layers.Activation('elu')(b1)

            c3 = layers.Conv2D(20, (3, 3), data_format='channels_last', padding='same')(elu1)
            b2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c3)

            con = layers.add([elu, b2])
            elu = layers.Activation('elu')(con)

        #c4 = layers.Conv2D(20, (18, 1), activation='elu', data_format='channels_last')(elu)

        p1 = layers.Permute((2, 1, 3))(elu)
        r1 = layers.Reshape((991, 22*20))(p1)
        mp1 = layers.AveragePooling1D(76, strides=15)(r1)

        f1 = layers.Flatten()(mp1)

        outputs = layers.Dense(4, activation='softmax', kernel_regularizer=L1(0.01), activity_regularizer=L2(0.01))(f1)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='cnn3')

        return model

    def test(self):
        inputs = layers.Input(shape=self.input_dim)

        r1 = layers.Reshape((100, 100, 1))(inputs)
        c1 = layers.Conv2D(20, (1, 3), data_format='channels_last')(r1)
        c1 = layers.Conv2D(20, (3, 1), data_format='channels_last')(c1)

        outputs = layers.Dense(4, activation='softmax', kernel_regularizer=L1(0.01), activity_regularizer=L2(0.01))(c1)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='cnn3')

        return model



if __name__ == "__main__":
    model = Model((100, 100), 20)
    rnn = model.test()
    rnn.summary()
