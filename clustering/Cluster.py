import pandas as pd
import numpy as np
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import AffinityPropagation
import time
import argparse
import os


def getMatrix(fps):
    nfps = len(fps)
    mat = np.ones((nfps, nfps))
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        mat[:i, i] = sims
        mat[i, :i] = sims

    return mat.astype('float16', copy=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-base-dir',
                        default=os.getcwd(),
                        help='Specify the directory for the output, defaults to the directory where '
                             'the script is executed (os.getcwd())')
    parser.add_argument('--input-file', required=True,
                        help='Specify the sdf file for which the clusters should be computed')
    parser.add_argument('--output-filename', default="Clusters.csv",
                        help='Specify how the output file containing the clusters should be named, default is Clusters.csv')
    parser.add_argument('--assign-folds',
                        action='store_true',
                        help='Use this flag if after the clustering clusters should be randomly assigned to one of 5 folds')

    input_args = parser.parse_args()
    ms = [x for x in Chem.SDMolSupplier(input_args.input_file) if x is not None]
    ids = [m.GetProp('_Name') for m in ms]
    print('No of mols:', len(ms))
    fps = [AllChem.GetMorganFingerprint(x, 2) for x in ms]
    print('No of fps:', len(fps))
    print('Getting Matrix')
    dist = getMatrix(fps)
    print('Mat_size', dist.shape)

    print('Clustering')
    af = AffinityPropagation(affinity='precomputed', verbose=True, copy=False, max_iter=1000, damping=0.9).fit(dist)
    cluster_centers_indices = af.cluster_centers_indices_

    n_clusters_ = len(cluster_centers_indices)

    print(af.labels_)

    out = pd.DataFrame({'ID': ids, 'Clusters': af.labels_})
    if input_args.assign_folds:
        out['Fold'] = 0
        clusts = np.unique(out['Clusters'])
        folds = [1, 2, 3, 4, 5]
        sizes = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
        for i in clusts:
            fold = np.random.choice(folds, 1)[0]
            ix = out['Clusters'] == i
            out.Fold[ix] = fold
            sizes[str(fold)] += sum(ix)
            if sizes[str(fold)] >= out.shape[0] / 5:
                x = np.where(folds == fold)[0][0]
                folds.pop(x)
            if len(folds) < 1:
                break

    savepath = os.path.join(input_args.out_base_dir, input_args.output_filename)
    out.to_csv(savepath)
    n = af.n_iter_
    sizes = np.unique(af.labels_, return_counts=True)

    with open(os.path.join(input_args.out_base_dir, 'Cluster.log'), 'a+') as f:
        f.writelines(['Clustering with scikitlearn Affinity propagation', time.strftime("%Y%m%d-%H%M%S"), '\n',
                      'Settings for Clustering:', '\n', str(af), '\n',
                      'Cluster sizes', str(sizes), '\n',
                      'No of clusters', str(len(af.cluster_centers_indices_)), '\n',
                      'Cluster center indices: ', str(af.cluster_centers_indices_), '\n',
                      'Cluster labels: ', str(af.labels_), '\n',
                      'Convergence iteration:', str(af.convergence_iter), '\n',
                      'Number of iterations:', str(n), '\n'])
