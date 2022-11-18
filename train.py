#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from util import plotGraph
from dataset import split_data, HdrVdpDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import QNet
import glob2
import re


#traing for a single epoch
def train(loader, model, optimizer, args):
    model.train()

    total_loss = 0.0
    counter = 0
    progress = tqdm(loader)

    for stim, q in progress:
        if torch.cuda.is_available():
            stim = stim.cuda()
            q = q.cuda()
        q_hat = model(stim)
        
        loss = F.mse_loss(q_hat, q)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        counter += 1
        
        progress.set_postfix({'loss': total_loss / counter})

    return total_loss / counter;
    
#evaluate for a single epoch
def eval(loader, model, optimizer, args):
    model.eval()

    total_loss = 0.0
    counter = 0
    progress = tqdm(loader)
    
    targets = []
    predictions = []
    
    for stim, q in progress:
        with torch.no_grad():
            if torch.cuda.is_available():
                stim = stim.cuda()
                q = q.cuda()
            q_hat = model(stim)
            loss = F.mse_loss(q_hat, q)
            
            total_loss += loss.item()
                        
        targets.append(q)
        predictions.append(q_hat)
        counter += 1
            
        progress.set_postfix({'loss': total_loss / counter})
        
    targets = torch.cat(targets, 0).squeeze()
    predictions = torch.cat(predictions, 0).squeeze()
    
    return (total_loss / counter), targets, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Q regressor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, help='Path to data dir')
    parser.add_argument('-g', '--group', type=int, help='grouping factor for augmented dataset')
    parser.add_argument('-gp', '--groupprecomp', type=int, default = 1, help='grouping type')
    parser.add_argument('-e', '--epochs', type=int, default=1024, help='Number of training epochs')
    parser.add_argument('-s', '--scaling', type=bool, default=False, help='scaling')
    parser.add_argument('-b', '--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-r', '--runs', type=str, default='runs/', help='Base dir for runs')
    parser.add_argument('--resume', default=None, help='Path to initial weights')
    
    args = parser.parse_args()

    args.groupprecomp = (args.groupprecomp == 1)
    ### Prepare run dir
    params = vars(args)
    params['dataset'] = os.path.basename(os.path.normpath(args.data))

    run_name = 'q_{0[dataset]}_lr{0[lr]}_e{0[epochs]}_b{0[batch]}'.format(params)
    run_dir = os.path.join(args.runs, run_name) 
    ckpt_dir = os.path.join(run_dir, 'ckpt')
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        os.makedirs(ckpt_dir)
    
    log_file = os.path.join(run_dir, 'log.csv')
    param_file = os.path.join(run_dir, 'params.csv')
    pd.DataFrame(params, index=[0]).to_csv(param_file, index=False)
    
    ### Load Data
    train_data, val_data, test_data = split_data(args.data, group=args.group, bPrecompGroup = args.groupprecomp)

    train_data.to_csv(os.path.join(run_dir, "train.csv"), ',')
    val_data.to_csv(os.path.join(run_dir, "val.csv"), ',')
    test_data.to_csv(os.path.join(run_dir, "test.csv"), ',')

    #create the loader for the training set
    train_data = HdrVdpDataset(train_data, args.data, args.group, bPrecompGroup = args.groupprecomp, bScaling = args.scaling)
    train_loader = DataLoader(train_data, shuffle=True,  batch_size=args.batch, num_workers=8, pin_memory=True)
    #create the loader for the validation set
    val_data = HdrVdpDataset(val_data, args.data, args.group, bPrecompGroup = args.groupprecomp, bScaling = args.scaling)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.batch, num_workers=8, pin_memory=True)
    #create the loader for the testing set
    test_data = HdrVdpDataset(test_data, args.data, args.group, bPrecompGroup = args.groupprecomp, bScaling = args.scaling)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)

    #create the model
    model = QNet()
    if(torch.cuda.is_available()):
        model = model.cuda()        

    #create the optmizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5, verbose=True)

    log = pd.DataFrame()
    
    #training loop
    best_mse = None
    a_t = []
    a_v = []
    a_te = []
    
    start_epoch = 1
    if args.resume:
       ckpt_dir_r = os.path.join(args.resume, 'ckpt')
       ckpts = glob2.glob(os.path.join(ckpt_dir_r, '*.pth'))
       assert ckpts, "No checkpoints to resume from!"
    
       def get_epoch(ckpt_url):
           s = re.findall("ckpt_e(\d+).pth", ckpt_url)
           epoch = int(s[0]) if s else -1
           return epoch, ckpt_url
    
       start_epoch, ckpt = max(get_epoch(c) for c in ckpts)
       print('Checkpoint:', ckpt)
       ckpt = torch.load(ckpt)
       model.load_state_dict(ckpt['model'])
       start_epoch = ckpt['epoch']
       best_mse = ckpt['mse_val']
       start_epoch = ckpt['epoch']
        
    for epoch in trange(start_epoch, args.epochs + 1):
        cur_loss = train(train_loader, model, optimizer, args)
        val_loss, targets, predictions = eval(val_loader, model, optimizer, args)
        test_loss, targets, predictions = eval(test_loader, model, optimizer, args)
       
        metrics = {'epoch': epoch}
        metrics['mse_train'] = cur_loss
        metrics['mse_val'] = val_loss
        metrics['mse_test'] = test_loss
        log = log.append(metrics, ignore_index=True)
        log.to_csv(log_file, index=False)
        
        a_t.append(cur_loss)
        a_v.append(val_loss)
        a_te.append(test_loss)

        if best_mse is None or (val_loss < best_mse):
            delta = (targets - predictions)
            errors = delta.cpu().numpy()
            pd.DataFrame(errors).to_csv(os.path.join(run_dir, 'errors_test.csv'))
            pd.DataFrame(errors).to_csv('errors_test.csv')

            plt.clf()
            sns.distplot(errors, kde=True, rug=True)
            plt.savefig(os.path.join(run_dir, 'hist_errors_test.png'))
            plt.savefig('hist_errors_test.png')

            plotGraph(a_t, a_v, a_te, '.', run_name)
            plotGraph(a_t, a_v, a_te, run_dir, run_name)
            best_mse = val_loss
            ckpt = os.path.join(ckpt_dir, 'ckpt_e{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'mse_train': cur_loss,
                'mse_val': val_loss,
                'mse_test': test_loss,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt)

        scheduler.step(val_loss)
