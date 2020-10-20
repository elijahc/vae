import os
import subprocess
from PIL import Image
import neptune

DEFAULT_DATA_ROOT='/home/elijahc/projects/vae'

proj = neptune.set_project('elijahc/DuplexAE')

exp = proj.get_experiments(id=['DPX-65'])[0]

def model_dir(exp=exp,data_root=DEFAULT_DATA_ROOT):

    mod_dir = os.path.join(*[
        data_root,
        exp.get_properties()['dir'],
        ])
    return mod_dir

def img_dir(exp=exp,data_root=DEFAULT_DATA_ROOT):
    return os.path.join(model_dir(),'recons')

def log_gif(exp=exp,data_root=DEFAULT_DATA_ROOT):
    gif_path=os.path.join(img_dir(),'recon_learning_optimized.gif')

    if os.path.exists(gif_path):
        exp.log_artifact(gif_path)
    else:
        raise ValueError('{} does not exist'.format(gif_path))

def create_gif(exp=exp,data_root=DEFAULT_DATA_ROOT):
    calling_dir = os.getcwd()
    os.chdir(img_dir(exp=exp))
    cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/elijahc/vae/master/scripts/create_gif.sh)"'

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    line_output = p.communicate()[0].decode().split('\n')

    for l in line_output:
        print(l)

    imgur_url = [l for l in line_output if l.startswith('http')][0]

    print('uploading gif as artifact\n')
    log_gif(exp=exp)
    os.chdir(calling_dir)
