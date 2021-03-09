from load_data import Data
from model import Model
from train import Train
from plot import Plot

if __name__ == "__main__":

    data = Data()
    data.setup(True)
    # data.data_filtering()
    # data.data_augment()
    # data.data_addNoise(0, 2)
    dataset = data.get_dataset()
    
    # show dataset in channels (visualize data preprocessing)
    # figure = Plot(dataset=dataset)
    # figure.show_dataset()

    models = Model(dataset[0].shape[1:], 50)
    m = models.cnn2()
    m.summary()

    train = Train(m, dataset, 50, 200, 'adam', 'sparse_categorical_crossentropy')
    train.training()
    train.evaluate()

    


# tensorboard --logdir logs/scalars --port=7000

