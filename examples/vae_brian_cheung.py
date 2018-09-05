from src.models import VAE_Builder,CheungVae
from src.data_loader import get_loader,prepare_keras_dataset,Shifted_Data_Loader
from src.config import get_config
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config, unparsed = get_config()
    print(config)

    DL = Shifted_Data_Loader(dataset=config.dataset)
    mod_builder = CheungVae(config.dataset)

    mod = Model(mod_builder.input,mod_builder.output)
    print(mod.summary())

