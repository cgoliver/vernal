""" All functions related to clustering.
Can be called from MAGA and visual_checks
"""

import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))


def optimize_silhouette(Z,
                        max_clusts=1000,
                        min_clusts=2,
                        clust_step=10,
                        plateau=2,
                        random_state=1,
                        save_name=None):
    """
        Return best number of clusters according to
        silhouette score.
    """
    from sklearn.metrics import silhouette_score

    best_sil = -1
    count = 0
    best_k = None
    sils = []
    for k in range(min_clusts, max_clusts, clust_step):
        print(f"Clustering on {k} clusters.")
        model = MiniBatchKMeans(n_clusters=k, random_state=random_state)
        print("DONE")
        labels = model.fit_predict(Z)
        sil = silhouette_score(Z,
                               labels,
                               sample_size=1000,
                               metric='euclidean')
        sils.append({'components': k, 'silhouette': sil})
        if count >= plateau:
            break
        elif sil > best_sil:
            print(f"new best {sil} on {k} components.")
            best_sil = sil
            best_k = k
            count = 0
        else:
            count += 1
    df = pd.DataFrame(sils)
    if save_name is not None:
        df.to_csv(save_name)
    return best_k


def optimize_bic(Z,
                 plateau=2,
                 max_clusts=1000,
                 min_clusts=2,
                 clust_step=10
                 ):
    best_bic = sys.maxsize
    best_k = None
    counter = 0
    bics = []
    for k in range(min_clusts, max_clusts, clust_step):
        gm = GaussianMixture(covariance_type='spherical', n_components=k)
        gm.fit(Z)
        bic = gm.bic(Z)
        bics.append({'components': k, 'bic': bic})
        print(f"current bic {bic}")
        if counter >= plateau:
            print("HIT PLATEAU")
            break
        elif bic < best_bic:
            print(f"new best {bic} on {k} components.")
            best_bic = bic
            best_k = k
            counter = 0
        else:
            counter += 1

    df = pd.DataFrame(bics)
    df.to_csv("bics.csv")

    return best_k


def gmm(Z,
        n_clusters=100,
        optimize=False,
        min_clusts=2,
        max_clusts=1000,
        clust_step=10,
        min_factor=1.01,
        random_state=None):
    if optimize:
        n_clusters = optimize_bic(Z,
                                  min_clusts=min_clusts,
                                  max_clusts=max_clusts,
                                  clust_step=clust_step
                                  )

    model = GaussianMixture(n_components=n_clusters,
                            covariance_type='spherical',
                            random_state=random_state)

    labels = model.fit_predict(Z)
    centers = model.means_
    scores = model.predict_proba(Z)

    return {'model': model,
            'labels': labels,
            'centers': centers,
            'spread': model.covariances_,
            'scores': scores,
            'n_components': len(set(labels)),
            'components': sorted(list(set(labels)))}


def som(Z,
        n=10,
        n_iter=20,
        batch_size=50, **kwargs):
    # SOMS
    import torch
    from tools.SOM import SOM

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Z_torch = torch.from_numpy(Z)
    Z_torch = Z_torch.to(device)

    m, n = 50, 50
    nsamples = Z.shape[0]
    dim = Z.shape[1]
    n_iter = 2
    batch_size = 50
    som = SOM(m, n, dim, n_iter, device=device, precompute=True, periodic=False)
    learning_error = som.fit(Z_torch, batch_size=batch_size)
    predicted_clusts, errors = som.predict_cluster(Z_torch)

    return {'model': som,
            'labels': predicted_clusts,
            'errors': errors}


def k_means_agg(centers, full_labels, distance_threshold=0.01):
    """
        Do k means.
    """

    from sklearn.cluster import AgglomerativeClustering

    agg = AgglomerativeClustering(distance_threshold=distance_threshold,
                                  n_clusters=None,
                                  linkage='single',
                                  affinity='euclidean')
    agg.fit(centers)
    meta_clusters = agg.labels_
    new_labels = np.array([meta_clusters[i] for i in full_labels])
    new_clust_ids = sorted(list(set(meta_clusters)))

    new_centers = []
    for i in new_clust_ids:
        # take old clusters assigned to current new cluster
        take_clusts = np.where(meta_clusters == i)
        old_centers = centers[take_clusts]
        center = np.mean(old_centers, axis=0)
        new_centers.append(center)
    new_centers = np.array(new_centers)

    return new_labels, new_centers


def k_means(Z,
            optimize=False,
            n_clusters=100,
            min_factor=1.05,
            min_clust=2,
            max_clusts=1000,
            clust_step=10,
            random_state=None,
            ):
    from sklearn.cluster import MiniBatchKMeans

    if optimize:
        n_clusters = optimize_silhouette(Z,
                                         min_factor=min_factor,
                                         max_clusts=max_clusts,
                                         min_clusts=min_clust,
                                         clust_step=clust_step)
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    model = model.fit(Z)

    def remove_redundant(kmean_model):
        """
        sometimes the kmeans will yield a model with collapsed centroids.

        This is to deal with this problem.
        """
        import scipy.spatial.distance as dist
        centers = kmean_model.cluster_centers_
        centers_pdist = dist.squareform(dist.pdist(centers))
        binary_centers_pdist = centers_pdist < 0.0001
        n = len(binary_centers_pdist)

        # Maybe not the most elegant, but I don't have a good idea :
        # Start from the last point, and if there is no previous point that is at a low distance, keep it
        ids_to_keep = set()
        for decreasing_i in range(n - 1, -1, -1):
            previous_neighbors = binary_centers_pdist[decreasing_i][:decreasing_i]
            if not previous_neighbors.any():
                ids_to_keep.add(decreasing_i)
        ids_to_keep = list(ids_to_keep)
        filtered_centers = centers[ids_to_keep]

        # Create a new object for consistent function semantics
        n_clusters = len(filtered_centers)
        pruned_model = MiniBatchKMeans(n_clusters=n_clusters, init=filtered_centers)
        pruned_model.cluster_centers_ = filtered_centers
        return pruned_model

    pruned_model = remove_redundant(kmean_model=model)
    clust_centers = pruned_model.cluster_centers_
    n_clusters = len(clust_centers)
    clust_ids = pruned_model.predict(Z)
    dists_to_center = []
    for i in range(n_clusters):
        dists_to_center.append(
            np.mean(list(map(partial(euclidean, clust_centers[i]), Z[np.where(clust_ids == i)]))))

    # assert len(dists_to_center) == len(set(clust_ids)), "Spread size doesnt match k"

    return {'model': pruned_model,
            'labels': clust_ids,
            'centers': clust_centers,
            'scores': pruned_model.transform(Z),
            'spread': dists_to_center,
            'n_components': len(set(clust_ids)),
            'components': sorted(list(set(clust_ids)))}


def cluster(Z, algo='k_means', **algo_params):
    """
    Clustering wrapper.
    Takes a data matrix and algo

    Returns cluster objects with similar scikit-api
    """

    if algo == 'k_means':
        clusters = k_means(Z, **algo_params)
    if algo == 'gmm':
        clusters = gmm(Z, **algo_params)
    if algo == 'som':
        clusters = som(Z, **algo_params)
    return clusters


if __name__ == "__main__":
    from tools.learning_utils import inference_on_list

    graph_dir = '../data/unchopped_v4_nr'
    model_output = inference_on_list('1hop_weight',
                                     graph_dir,
                                     os.listdir(graph_dir),
                                     max_graphs=8000,
                                     nc_only=True
                                     )

    Z = model_output['Z']

    clust_info = cluster(Z,
                         algo='k_means',
                         optimize=True,
                         )

    pass
