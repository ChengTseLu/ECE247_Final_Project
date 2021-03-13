import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.optimizers import Adam
import numpy as np

from load_data import Data
from plot import Plot

class GAN:
    def __init__(self, dataset):    
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    print('-------------------------')
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.Xtrain, self.ytrain = dataset[0], dataset[1]
        self.Xval, self.yval = dataset[2], dataset[3]
        
        self.G = self.generator()
        self.D = self.discriminator()
        self.G_and_D = self.combine()

        self.D.compile(Adam(0.00001), 'binary_crossentropy')
        self.G_and_D.compile('adam', 'binary_crossentropy')

    def generator(self):
        
        inputs = layers.Input(shape=(125, 1))

        r1 = layers.Reshape((125, 1, 1))(inputs)
        
        # TC1 = layers.Conv2DTranspose(256, (3, 3), (1, 1), data_format='channels_first', padding='same')(inputs)

        TC1 = layers.Conv2DTranspose(64, (3, 3), (1, 1), activation=LeakyReLU(), padding='same')(r1)

        TC1 = layers.Conv2DTranspose(32, (3, 3), (2, 2), activation=LeakyReLU(), padding='same')(TC1)

        TC1 = layers.Conv2DTranspose(22, (3, 3), (2, 1), activation=LeakyReLU(), padding='same')(TC1)

        p1 = layers.Permute((3, 1, 2))(TC1)
        r1 = layers.Reshape((22, 1000))(p1)

        model = models.Model(inputs=inputs, outputs=r1, name='generator')

        model.summary()

        return model

    def discriminator(self):

        inputs = layers.Input((22, 1000))

        r1 = layers.Reshape((22, 1000, 1))(inputs)

        c1 = layers.Conv2D(22, (3, 3), activation=LeakyReLU())(r1)
        b1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c1)

        c2 = layers.Conv2D(32, (3, 3), activation=LeakyReLU())(b1)
        b2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c2)

        c3 = layers.Conv2D(64, (3, 3), activation=LeakyReLU())(b2)
        b3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c3)

        c4 = layers.Conv2D(128, (3, 3), activation=LeakyReLU())(b3)
        b4 = layers.BatchNormalization(momentum=0.9, epsilon=1e-05)(c4)

        # p1 = layers.Permute((2, 1, 3))(b4)
        # r1 = layers.Reshape((985, -1))(p1)
        # mp1 = layers.AveragePooling1D(72, strides=15)(r1)
        f1 = layers.Flatten()(b4)

        dp = layers.Dropout(0.1)(f1)

        outputs = layers.Dense(1, activation='sigmoid')(dp)

        model = models.Model(inputs=inputs, outputs=outputs, name='discriminator')

        model.summary()

        return model

    def combine(self):
        self.D.trainable = False

        model = models.Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, epochs=500, batch=16):

        figure = Plot()
        fix_noise = np.random.normal(0, 5, (batch, 125, 1))
        for i in range(epochs):

            # train discriminator
            random_index = np.random.randint(0, len(self.Xtrain), size=batch)
            gt_data = self.Xtrain[random_index]
            
            gen_noise = np.random.normal(0, 5, (batch, 125, 1))
            fake_data = self.G.predict(gen_noise)

            x_combined_batch = np.concatenate((gt_data, fake_data))
            y_combined_batch = np.concatenate((np.ones((batch, 1)), np.zeros((batch, 1))))

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

            # train generator
            noise = np.random.normal(0, 5, (batch, 125, 1))
            y_gen_label = np.ones((batch, 1))

            g_loss = self.G_and_D.train_on_batch(noise, y_gen_label)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (i, d_loss, g_loss))

            if i%100 == 0:
                figure.show_dataset(self.G.predict(fix_noise))
            
            






if __name__ == "__main__":
    data = Data()
    data.setup(True)
    data.plot_dataset()
    dataset = data.get_dataset()

    gan = GAN(dataset)
    #gan.train()