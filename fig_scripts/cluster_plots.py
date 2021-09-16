import sys
import os
from collections import Counter
import pickle
from scipy.spatial.distance import cdist

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from tools.learning_utils import inference_on_list
from tools.clustering import *

from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


def get_n_clusts(run, graph_dir, save_name="silhouette.csv"):
    model_output = inference_on_list(run,
                                     graph_dir,
                                     graph_list=os.listdir(graph_dir),
                                     )
    Z = model_output['Z']
    best_number = optimize_silhouette(Z,
                                      max_clusts=500,
                                      min_clusts=2,
                                      clust_step=5,
                                      plateau=4,
                                      random_state=1,
                                      save_name=save_name
                                      )
    print('best number of clusters is : ', best_number)
    return best_number


def plot_silhouette(csv_file):
    """
    Take as input the csv file produced by get_nclusts
    """
    df = pd.read_csv(csv_file)
    components = df['components']
    silhouette = df['silhouette']
    plt.plot(components, silhouette)
    plt.xlabel('Number of Components')
    plt.ylabel('Silhouette Score')
    plt.savefig('../figs/silhouette_score.pdf')
    plt.show()


def get_clusters_plots(run, graph_dir, n_clusters):
    recompute = True
    if recompute:
        model_output = inference_on_list(run,
                                         graph_dir,
                                         graph_list=os.listdir(graph_dir),
                                         )
        Z = model_output['Z']
        model = MiniBatchKMeans(n_clusters=n_clusters)
        labels = model.fit_predict(Z)
        centroids = model.cluster_centers_
        Z = Z[np.random.choice(len(Z), size=10000, replace=False)]
        pickle.dump((Z, centroids, labels), open('clusters_plotter.p', 'wb'))
    Z, centroids, labels = pickle.load(open('clusters_plotter.p', 'rb'))

    # Print mean distance
    pdist_z = pdist(Z)
    plt.hist(pdist_z, range=(0, 2))
    plt.xlabel('Pairwise distance between embeddings')
    plt.ylabel('Counts')
    plt.savefig('../figs/pdist_embeddings.pdf')
    plt.show()

    # Print populations
    counter = Counter(labels)
    populations = list(reversed(sorted(counter.values())))
    # populations = [pop if pop < 200 else 200 for pop in populations]
    # print(populations)
    plt.plot(range(len(populations)), populations)
    plt.xlabel('Sorted Cluster ID')
    plt.ylabel('Cluster Population (thresholded)')
    plt.yscale('log')
    plt.savefig('../figs/cluster_populations.pdf')
    plt.show()

    # Print distance to centroid
    cdist_z = cdist(Z, centroids)
    min_dists = np.min(cdist_z, axis=1)
    plt.hist(min_dists, range=(0, 1.75))
    plt.xlabel('Distance to Centroid')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.savefig('../figs/cdist_embeddings.pdf')
    plt.show()


if __name__ == "__main__":
    n_hops = 1
    csv_name = f'../results/trained_models/silhouette_{n_hops}hop.csv'
    get_n_clusts(f'new_kernel_{n_hops}', graph_dir='../data/graphs/NR/', save_name=csv_name)
    plot_silhouette(csv_file=csv_name)
    get_clusters_plots(f'new_kernel_{n_hops}', graph_dir='../data/graphs/NR/', n_clusters=200)
