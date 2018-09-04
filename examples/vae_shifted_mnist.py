from src.models import VAE_Builder
from src.data_loader import get_loader,prepare_keras_dataset,Shifted_Data_Loader
from src.config import get_config
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

def plot_sample_image(DL,i=250):
    class_ids = np.unique(DL.y_train)
    masks_train = [DL.y_train==i for i in class_ids]
    masks_test = [DL.y_test==i for i in class_ids]

    print(DL.y_train[masks_train[2]][i])
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(DL.x_train[masks_train[2]][i].reshape(28,28))
    axs[1].imshow(DL.sx_train[masks_train[2]][i].reshape(28*2,28*2))


if __name__ == "__main__":
    config, unparsed = get_config()
    print(config)

    DL = Shifted_Data_Loader(dataset=config.dataset)

    mod_builder = VAE_Builder()
    mod_builder.build(input_shape=(4*784,))
    mod = Model(mod_builder.input,mod_builder.output)
    print(mod.summary())