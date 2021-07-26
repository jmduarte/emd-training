import awkward as ak
import pandas as pd
import numpy as np
import torch
import math

from coffea.nanoevents.methods import vector
from pyjet import cluster,DTYPE_PTEPM

ak.behavior.update(vector.behavior)

def jet_particles(raw_path, n_events, back):
    if back:
        start = 1e6 - n_events
        df = pd.read_hdf(raw_path, start=start)
    else:
        df = pd.read_hdf(raw_path, stop=n_events)
    all_events = df.values
    rows = all_events.shape[0]
    cols = all_events.shape[1]
    X = []
    # cluster jets and store info
    for i in range(rows):
        pseudojets_input = np.zeros(len([x for x in all_events[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
        for j in range(cols // 3):
            if (all_events[i][j*3]>0):
                pseudojets_input[j]['pT'] = all_events[i][j*3]
                pseudojets_input[j]['eta'] = all_events[i][j*3+1]
                pseudojets_input[j]['phi'] = all_events[i][j*3+2]
        sequence = cluster(pseudojets_input, R=1.0, p=-1)
        jets = sequence.inclusive_jets()[:2] # leading 2 jets only
        if len(jets) < 2: continue
        for jet in jets: # for each jet get (px, py, pz, e)
            if jet.pt < 200 or len(jets)<=1: continue
            n_particles = len(jet)
            particles = np.zeros((n_particles, 3))
            # store all the particles of this jet
            for p, part in enumerate(jet):
                particles[p,:] = np.array([part.pt,
                                           part.eta,
                                           part.phi])
            X.append(particles)
    X = np.array(X,dtype='O')
    return X

def normalize(jet):
    # convert into a coffea vector
    part_vecs = ak.zip({
        "pt": jet[:, 0:1],
        "eta": jet[:, 1:2],
        "phi": jet[:, 2:3],
        "mass": np.zeros_like(jet[:, 1:2])
        }, with_name="PtEtaPhiMLorentzVector")

    # sum over all the particles in each jet to get the jet 4-vector
    jet_vecs = part_vecs.sum(axis=0)

    # subtract the jet eta, phi from each particle to convert to normalized coordinates
    jet[:, 1] -= jet_vecs.eta.to_numpy()
    jet[:, 2] -= jet_vecs.phi.to_numpy()

    # divide each particle pT by jet pT if we want relative jet pT
    jet[:, 0] /= jet_vecs.pt.to_numpy()
    
def remove_dupes(data):
    """
    remove duplicate data with alternative jet orderings
    """
    n_jets = int(math.sqrt(len(data)))
    pairs = []
    for r in range(n_jets):
        for c in range(r, n_jets):
            r_idx = n_jets * r
            if c == r:
                pairs.append(data[r_idx + c])
                if data[r_idx + c].y.item() != 0:
                    exit("EMD non-zero")
            else:
                d1 = data[r_idx + c]
                d2 = data[c * n_jets + r]
                pairs.append(d2)
                if d1.y.item() != d2.y.item():
                    exit("Unexpected non-dupe")
    return pairs

def pair_dupes(data):
    """
    pair duplicate data with alternative jet orderings
    """
    n_jets = int(math.sqrt(len(data)))
    pairs = []
    for r in range(n_jets):
        for c in range(r, n_jets):
            r_idx = n_jets * r
            if c == r:
                if data[r_idx + c].y.item() != 0:
                    exit("EMD non-zero")
            else:
                d1 = data[r_idx + c]
                d2 = data[c * n_jets + r]
                pairs.append([d1, d2])
                if d1.y.item() != d2.y.item():
                    exit("Unexpected non-dupe")
    return pairs
