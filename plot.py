import matplotlib.pyplot as plt

class Plot:
    def __init__(self, loss_hist):
        self.hist = loss_hist.history

    def draw(self):
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.plot(self.hist['loss'])
        plt.plot(self.hist['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])

        plt.subplot(1, 2, 2)
        plt.plot(self.hist['acc'])
        plt.plot(self.hist['val_acc'])
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])

        plt.show()