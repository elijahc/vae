from src.models import VAE_Builder
from src.data_loader import prepare_keras_dataset
from src.config import get_config
from keras.datasets import mnist
from keras.models import Model


if __name__ == "__main__":
    config, unparsed = get_config()
    print(config)

    mod_builder = VAE_Builder()
    mod_builder.build(input_shape=(784,))
    mod = Model(mod_builder.input,mod_builder.output)
    print(mod.summary())
