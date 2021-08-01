import os
import os.path as osp
import torch
import itertools
import tables
import numpy as np
import energyflow as ef
import glob
import logging

from torch_geometric.data import Dataset, Data
from process_util import jet_particles, normalize
from natsort import natsorted
from sys import exit

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, n_jets=1000,
                 n_events_merge=100, n_events=1000, lhco=False, lhco_back=False, R=1.0):
        self.n_jets = n_jets
        self.n_events_merge = n_events_merge
        self.n_events = n_events
        self.lhco = lhco
        self.lhco_back = lhco_back
        self.R = R
        super(GraphDataset, self).__init__(root, transform, pre_transform) 


    @property
    def raw_file_names(self):
        return ['events_LHCO2020_backgroundMC_Pythia.h5']

    @property
    def processed_file_names(self):
        """
        Returns a list of all the files in the processed files directory
        """
        proc_list = glob.glob(osp.join(self.processed_dir, 'data_*.pt'))
        n_files = int(self.n_jets*self.n_jets/self.n_events_merge)
        return_list = list(map(osp.basename, proc_list))[:n_files]
        return natsorted(return_list)

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        Js = []
        for raw_path in self.raw_paths:
            # load jet-particles dataset
            if self.lhco or self.lhco_back:
                print("Loading LHCO Dataset")
                X = jet_particles(raw_path, self.n_events, self.lhco_back, self.R)
            else:
                print("Loading QG Dataset")
                X, _ = ef.qg_jets.load(self.n_jets, pad=False, cache_dir=self.root+'/raw')
            
            # store list of jets as particles (pt_rel, eta_rel, phi_rel)
            Js = []
            for jet_ctr, x in enumerate(X): 
                x = normalize(x) # pt_rel, phi_rel, eta_rel
                Js.append(x)
                # stop when n_jets stored
                if jet_ctr == self.n_jets: break

        # calc emd between all jet pairs and save datum
        jetpairs = [[i, j] for (i, j) in itertools.product(range(self.n_jets),range(self.n_jets))]
        datas = []
        for k, (i, j) in enumerate(jetpairs):    
            if k % (len(jetpairs) // 20) == 0:
                print(f'Generated: {k}/{len(jetpairs)}')
            emdval, G = ef.emd.emd(Js[i], Js[j], R=self.R, return_flow=True)

            # differentiate 2 jets by column of 1 vs -1
            ji = np.zeros((Js[i].shape[0],Js[i].shape[1]+1))
            jj = np.zeros((Js[j].shape[0],Js[j].shape[1]+1))
            ji[:,:3] = Js[i].copy()
            jj[:,:3] = Js[j].copy()
            ji[:,3] = -1*np.ones((Js[i].shape[0]))
            jj[:,3] = np.ones((Js[j].shape[0]))
            jetpair = np.concatenate([ji, jj], axis=0)

            nparticles_i = len(Js[i])
            nparticles_j = len(Js[j])
            pairs = [[m, n] for (m, n) in itertools.product(range(0,nparticles_i),range(nparticles_i,nparticles_i+nparticles_j))]
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            edge_y = torch.tensor([[G[m,n-nparticles_i] for m, n in pairs]], dtype=torch.float)
            edge_y = edge_y.t().contiguous()

            x = torch.tensor(jetpair, dtype=torch.float)
            y = torch.tensor([[emdval]], dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, y=y, edge_y=edge_y)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            datas.append([data])                  
            if k%self.n_events_merge == self.n_events_merge-1:
                datas = sum(datas,[])
                torch.save(datas, osp.join(self.processed_dir, 'data_{}.pt'.format(k)))
                datas=[]
            
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="Directory to process data", required=False,
                        default='/energyflowvol/datasets/')
    parser.add_argument("--lhco", action='store_true', help="Using lhco dataset (diff processing)", default=False, required=False)
    parser.add_argument("--lhco-back", action="store_true", help="Start from tail end of lhco data to get unused dataset", required=False)
    parser.add_argument("--n-jets", type=int, help="number of jets", required=False, default=100)
    parser.add_argument("--n-events-merge", type=int, help="number of events to merge", required=False, default=1)
    args = parser.parse_args()

    os.makedirs(args.input_dir,exist_ok=True)

    # log arguments
    logging.basicConfig(filename=osp.join(args.input_dir, "logs.log"), filemode='w', level=logging.DEBUG, format='%(asctime)s | %(levelname)s: %(message)s')
    for arg, value in sorted(vars(args).items()):
            logging.info("Argument %s: %r", arg, value)

    gdata = GraphDataset(root=args.input_dir, n_jets=args.n_jets, n_events_merge=args.n_events_merge, lhco=args.lhco, lhco_back=args.lhco_back)

    print("Done")
