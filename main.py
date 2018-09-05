import numpy as np
import tensorflow as tf
from src.utils import prepare_dirs_and_logger
from src.data_loader import get_loader,Shifted_Data_Loader
from src.models import TandemVAEBuilder

from src.config import get_config

def main(config):
    DL = Shifted_Data_Loader(config.dataset)

    builder = TandemVAEBuilder(config.enc_layers,config.y_dim,config.z_dim)
    builder.build(input_shape=(784*4,))

    import pdb; pdb.set_trace()
    prepare_dirs_and_logger(config)

if __name__ == "__main__":
    config, unparsed = get_config()

    main(config)
