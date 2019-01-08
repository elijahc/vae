#-*- coding: utf-8 -*-
import argparse
import json

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--enc_layers', nargs='+', type=int, default=[500,500],
                     help='network encoder arcitecture')
net_arg.add_argument('--y_dim', type=int, default=10,
                     help='number of units to represent identity')
net_arg.add_argument('--z_dim', type=int, default=2,
                     help='number of units to represent non-identity')
net_arg.add_argument('--recon', type=int, default=1,
                     help='weight of the reconstruction loss term')
net_arg.add_argument('--xcov', type=int, default=10,
                     help='weight of the xcov loss term')
net_arg.add_argument('--xent', type=int, default=10,
                     help='weight of the xentropy loss term')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='mnist')
# data_arg.add_argument('--shifted'
data_arg.add_argument('--batch_size', type=int, default=32)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--epochs', type=int, default=5)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_level', type=str, default='INFO',
                    choices=['INFO','DEBUG','WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')

def get_config():
    config, unparsed = parser.parse_known_args()

    # Set or fix any inferences from params here

    return config, unparsed

def load_config(model_dir):
    fp = os.path.join(model_dir,'params.json')
    print('loading...',fp)
    with open(fp, 'r') as f:
        json_config = json.load(f)
        config = argparse.Namespace()
        for k in json_config.keys():
            setattr(config, k, json_config[k])
        return config