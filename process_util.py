import numpy as np
import pandas as pd
from pyjet import cluster,DTYPE_PTEPM

def jet_particles(raw_path, n_events):
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