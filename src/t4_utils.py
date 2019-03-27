import t4
import pandas as pd
import numpy as np
import os
import datetime as dt
import json

def get_package(slug,buk='jzlab'):
    p = t4.Package.browse(slug,'s3://{}'.format(buk))

    print(p)
    return p

def stage_export_model(dir_path,commit_message=None,files=None):
    p = get_package('elijahc/vae')

    if files is None:
        files = ['weights.h5',
                 'model.json',
                 'config.json',
                 'train_history.parquet',
                 ]

    with open(os.path.join(dir_path,'config.json'), 'r') as fp:
        run_meta = json.load(fp)

    for fn in files:
        fp = os.path.join(dir_path,fn)
        if os.path.exists(fp):
            p = p.set(fp,fp,meta=run_meta)

    return p
