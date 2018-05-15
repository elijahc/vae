import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_transcoder

def example_results(test_set,encoder,decoder,n=10,shuffle=False):
    transc = get_transcoder(encoder,decoder)
    if shuffle is True:
        np.random.shuffle(test_set)

    im_true = test_set[np.arange(n)]
    decoded_imgs = transc(im_true)

    plt.figure(figsize=(20, 4))
    for i in np.arange(n):
        x_true = im_true[i]
        x_pred = decoded_imgs[i]
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_true.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_pred.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()