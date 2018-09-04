#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
# net_arg = add_argument_group('Network')
# net_arg.add_argument('--input_scale_size', type=int, default=64,
#                      help='input image will be resized with the given value as width and height')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='mnist')
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