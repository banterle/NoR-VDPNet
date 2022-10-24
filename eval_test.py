import os
import re
import glob2
import argparse
import pandas as pd

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import HdrVdpDataset
from model import QNet

def predict(loader, model, args):
    model.eval()
    
    targets = []
    predictions = []
    progress = tqdm(loader)
    for stim, q in progress:
        with torch.no_grad():
            stim = stim.cuda()
            q_hat = model(stim)
            targets.append(q.data)
            predictions.append(q_hat.data.cpu())

    targets = torch.cat(targets, 0).squeeze()
    predictions = torch.cat(predictions, 0).squeeze()
    
    return targets, predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Q regressor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run', type=str, help='Base dir of run to evaluate')
    parser.add_argument('data', type=str, help='Path to data dir')
    parser.add_argument('out', type=str, help='Path to out dir')
    args = parser.parse_args()
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    ckpt_dir = os.path.join(args.run, 'ckpt')
    log_file = os.path.join(args.run, 'log.csv')
    param_file = os.path.join(args.run, 'params.csv')
    
    ### Load Data
    data = os.path.join(args.run, 'test.csv')
    test_data = pd.read_csv(data)
        
    test_data = HdrVdpDataset(test_data, args.data)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)

    ### Create Model
    model = QNet().cuda()
    
    ckpts = glob2.glob(os.path.join(ckpt_dir, '*.pth'))
    assert ckpts, "No checkpoints to resume from!"

    def get_epoch(ckpt_url):
        s = re.findall("ckpt_e(\d+).pth", ckpt_url)
        epoch = int(s[0]) if s else -1
        return epoch, ckpt_url

    start_epoch, ckpt = max(get_epoch(c) for c in ckpts)
    print('Checkpoint:', ckpt)
    ckpt = torch.load(ckpt)
    model.load_state_dict(ckpt['model'])
    
    targets, predictions = predict(test_loader, model, args)
    delta = (targets - predictions)
    errors = delta.cpu().numpy()
    pd.DataFrame(errors).to_csv(os.path.join(args.run, "errors.csv"))
    pd.DataFrame(errors).to_csv(os.path.join(args.out, "errors.csv"))

    pd.DataFrame(predictions.cpu().numpy()).to_csv(os.path.join(args.run, "predictions.csv"))
    pd.DataFrame(targets.cpu().numpy()).to_csv(os.path.join(args.run, "targets.csv"))
    sns.distplot(errors, kde=True, rug=True)
    hist_file = os.path.join(args.run, 'hist.png')
    plt.savefig(hist_file)
    plt.savefig(os.path.join(args.out, 'hist.png'))

