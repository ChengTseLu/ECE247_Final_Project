from load_data import Data
from model import Model
from train import Train


if __name__ == "__main__":

    data = Data()
    data.setup(True)
    # data.data_filtering()
    # data.data_augment()
    # data.data_addNoise(0, 2)
    # data.plot_dataset()
    dataset = data.get_dataset()
    

    models = Model(dataset[0].shape[1:], 50)
    m = models.cnn2()
    m.summary()

    train = Train(m, dataset, 50, 50, 'adam', 'sparse_categorical_crossentropy')
    train.training()
    train.evaluate()

    


# tensorboard --logdir logs/scalars --port=7000

