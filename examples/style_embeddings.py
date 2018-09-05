import numpy as np
from sklearn.manifold import Isomap
from src.data_loader import prepare_keras_dataset
from src.utils import gen_sorted_isomap
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm

(x_train,x_test),(y_train,y_test) = prepare_keras_dataset(fashion_mnist)
y_masks = [np.where(y_test==i)[0] for i in np.arange(10)]

iso_sorted = []
for cls_id in tqdm(np.arange(10)):
    iso_sorted.append(gen_sorted_isomap(y_train[y_masks[cls_id]],np.arange(1000),n_neighbors=150,n_components=1))

def plot_first(ims,n=10):
    fig, axs = plt.subplots(1,n)

    for ax,im in zip(axs,ims[:n]):
        ax.imshow(im.reshape(28,28))
