from load_data import Data
from model import Model
from train import Train
from plot import Plot

if __name__ == "__main__":
    data = Data()
    data.setup(True)
    dataset = data.get_dataset()

    models = Model(dataset[0].shape[1:], 50)
    m = models.rnn_LSTM()
    m.summary()

    train = Train(m, dataset, 50, 300, 'adam', 'sparse_categorical_crossentropy')
    train.training()
    train.evaluate()


# tensorboard --logdir logs/scalars --port=7000

