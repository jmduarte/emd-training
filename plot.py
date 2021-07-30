'''
File: plot.py
Plot either input distributions or graphs of emd-network output

Example (visualize model output):
python plot.py --plot-nn-eval \
    --model 'SymmetricDDEdgeNet' \
    --data-dir '/energyflowvol/data/eval_lhco_data_150' \
    --save-dir '/energyflowvol/figures/symmDD_lhco_model_new_loss_1k' \
    --model-dir '/energyflowvol/models/symmDD_lhco_model_1k_new_loss' \
    --n-jets 150 \
    --n-events-merge 500 \
    --remove-dupes
'''
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import inspect
import torch
import math
import tqdm
from pathlib import Path
from torch.utils.data import random_split
from graph_data import GraphDataset
from torch_geometric.data import Data, DataLoader

# personal code
import models
from process_util import remove_dupes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def make_hist(data, label, save_dir):
    plt.figure(figsize=(6,4.4))
    plt.hist(data)
    plt.legend()
    plt.xlabel(label, fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, label+'.pdf'))
    plt.close()

def get_y_output(gdata):
    y = []
    for d in gdata:
        y.append(d[0].y[0])
    y = torch.cat(y)
    return y

def get_x_input(gdata):
    pt = []; eta = []; phi = []
    for d in gdata:
        pt.append(d[0].x[:,0])
        eta.append(d[0].x[:,1])
        phi.append(d[0].x[:,2])
    pt = torch.cat(pt)
    eta = torch.cat(eta)
    phi = torch.cat(phi)
    return (pt, 'pt'), (eta, 'eta'), (phi, 'phi')

def make_plots(preds, ys, model_fname, save_dir):

    # largest y-value rounded to nearest 100
    max_range = max(np.max(ys), np.max(preds))
    
    diffs = (preds-ys)
    rel_diffs = diffs[ys>0]/ys[ys>0]

    # plot figures
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(ys, bins=np.linspace(0, max_range , 101),label='True', alpha=0.5)
    plt.hist(preds, bins=np.linspace(0, max_range, 101),label = 'Pred.', alpha=0.5)
    plt.legend()
    ax.set_xlabel('EMD [GeV]') 
    fig.savefig(osp.join(save_dir,model_fname+'_EMD.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    hts, bins, _ = plt.hist(diffs, bins=np.linspace(-0.1, 0.1, 101))
    ax.set_xlabel(f'EMD diff. [GeV]')
    x = max(bins) * 0.3
    y = max(hts) * 0.8
    mu = np.format_float_scientific(np.mean(diffs), precision=3)
    sigma = np.format_float_scientific(np.std(diffs), precision=3)
    plt.text(x, y, f'$\mu={mu}$'
                '\n'
                f'$\sigma={sigma}$')
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_diff.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_diff.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    hts, bins, _ = plt.hist(rel_diffs, bins=np.linspace(-1, 1, 101))
    ax.set_xlabel(f'EMD rel. diff.') 
    x = max(bins) * 0.3
    y = max(hts) * 0.8
    mu = np.format_float_scientific(np.mean(rel_diffs), precision=3)
    sigma = np.format_float_scientific(np.std(rel_diffs), precision=3)
    plt.text(x, y, f'$\mu={mu}$'
                '\n'
                f'$\sigma={sigma}$')
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_rel_diff.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_rel_diff.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    x_bins = np.linspace(0, max_range, 101)
    y_bins = np.linspace(0, max_range, 101)
    plt.hist2d(ys, preds, bins=[x_bins,y_bins])
    ax.set_xlabel('True EMD [GeV]')  
    ax.set_ylabel('Pred. EMD [GeV]')
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_corr.pdf'))
    fig.savefig(osp.join(save_dir,model_fname+'_EMD_corr.png'))


if __name__ == '__main__':
    import argparse;
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-input', action='store_true', help='plot pt eta phi', default=False, required=False)
    parser.add_argument('--plot-nn-eval', action='store_true', help='plot graphs for evaluating emd nn', default=False, required=False)
    parser.add_argument('--model', choices=[m[0] for m in inspect.getmembers(models, inspect.isclass) if m[1].__module__ == 'models'], 
                        help='Model name', required=False, default='DeeperDynamicEdgeNet')
    parser.add_argument('--data-dir', type=str, help='location of dataset', default='~/.energyflow/datasets', required=True)
    parser.add_argument('--save-dir', type=str, help='where to save figures', default='/energyflowvol/figures', required=True)
    parser.add_argument('--model-dir', type=str, help='path to folder with model', default='/energyflowvol/models/', required=False)
    parser.add_argument('--n-jets', type=int, help='number of jets', required=False, default=150)
    parser.add_argument('--n-events-merge', type=int, help='number of events to merge', required=False, default=500)
    parser.add_argument("--batch-size", type=int, help="batch size", required=False, default=64)
    parser.add_argument('--remove-dupes', action='store_true', help='remove dupes in data with different jet ordering', required=False)
    parser.add_argument("--lhco", action='store_true', help="Using lhco dataset (diff processing)", default=True, required=False)
    parser.add_argument("--lhco-back", action='store_true', help="generate data from tail end of raw data", default=True, required=False)
    args = parser.parse_args()

    Path(args.save_dir).mkdir(exist_ok=True) # make a folder for these graphs
    gdata = GraphDataset(root=args.data_dir, n_jets=args.n_jets, n_events_merge=args.n_events_merge, lhco=args.lhco, lhco_back=args.lhco_back)

    if args.plot_input:
        x_input = get_x_input(gdata)
        for d in x_input:
            data = d[0]; label = d[1]
            make_hist(data.numpy(), label, args.save_dir)

    if args.plot_nn_eval:
        if args.model_dir is None:
            exit('No args.model-dir not specified')

        # load all data into memory at once
        test_dataset = []
        for g in gdata:
            test_dataset += g
        if args.remove_dupes:
            test_dataset = remove_dupes(test_dataset)

        # load in model
        input_dim = 3
        big_dim = 32
        bigger_dim = 128
        global_dim = 2
        output_dim = 1
        batch_size=args.batch_size
        model_class = getattr(models, args.model)
        model = model_class(input_dim=input_dim, big_dim=big_dim, bigger_dim=bigger_dim, 
                            global_dim=global_dim, output_dim=output_dim).to(device)
        model_fname = args.model
        modpath = osp.join(args.model_dir,model_fname+'.best.pth')
        try:
            print(f'Loading model from: {modpath}')
            model.load_state_dict(torch.load(modpath, map_location=device))
        except:
            exit('No model')
        
        # get test dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        test_samples = len(test_dataset)
        
        # save folder
        eval_folder = 'eval'
        if not args.remove_dupes:
            eval_folder += '_dupes'
        eval_dir = osp.join(args.save_dir, eval_folder)
        Path(eval_dir).mkdir(exist_ok=True)

        # evaluate model
        ys = []
        preds = []
        diffs = []
        t = tqdm.tqdm(enumerate(test_loader),total=test_samples/batch_size)
        model.eval()
        for i, data in t:
            data.to(device)
            out = model(data)
            if 'SymmetricDDEdgeNet' in model_fname:
                out = out[0]    # toss unecessary terms
            ys.append(data.y.cpu().numpy().squeeze())
            preds.append(out.cpu().detach().numpy().squeeze())
        ys = np.concatenate(ys)   
        preds = np.concatenate(preds)   

        # plot results
        make_plots(preds, ys, model_fname, eval_dir)
