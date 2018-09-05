import numpy as np
import tensorflow as tf
from src.utils import prepare_dirs_and_logger
from src.data_loader import get_loader,Shifted_Data_Loader
from src.models import TandemVAEBuilder
from src.trainer import Trainer
from src.utils import ElasticSearchMonitor
from src.losses import acc, 
from keras.callbacks import TerminateOnNaN

from src.config import get_config

if __name__ == "__main__":
    config, unparsed = get_config()

    prepare_dirs_and_logger(config)

    DL = Shifted_Data_Loader(config.dataset)

    trainer = Trainer(config,DL)

    mod = trainer.model
    es_logger = ElasticSearchMonitor(root='http://localhost:9200',path='/tensorflow')
    ToN = TerminateOnNaN()

    trainer.go(trainer.data_loader.sx_train,trainer.data_loader.y_train_oh,
        validation_split=0.1,
        shuffle=True,
        callbacks=[es_logger,ToN],
        verbose=0)

    import ipdb; ipdb.set_trace()
    # main(config)
