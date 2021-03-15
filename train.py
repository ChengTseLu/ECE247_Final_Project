from tensorflow import keras
from plot import Plot
import time
import datetime


class Train:
    def __init__(self, model, dataset, testset, epochs, batch_size, opt, loss):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.Xtrain, self.ytrain = dataset[0], dataset[1]
        self.Xval, self.yval = dataset[2], dataset[3]
        self.testset = testset
        self.logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model.compile(opt, loss, metrics=['acc'])

    def training(self):
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.loss_hist = self.model.fit(self.Xtrain, self.ytrain
                                        , batch_size=self.batch_size
                                        , validation_data=(self.Xval, self.yval)
                                        , epochs=self.epochs
                                        , callbacks=[self.tensorboard_callback])

    def evaluate(self):
        score = self.model.evaluate(self.testset[0], self.testset[1], verbose=0)
        print('Test Accuracy: ', score[1])
        figure = Plot(self.loss_hist)
        figure.show_loss()

    def get_loss_history(self):
        return self.tensorboard_callback, self.loss_hist