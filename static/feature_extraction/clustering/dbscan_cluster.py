from sklearn.cluster import DBSCAN
import numpy as np

# Euclidean Distance Caculator using Forbenius Norm 
def FD_euc_dist(a, b, ax=1):
  """
  Input - two Fourier Descriptors, in np array form (output from final_FD())
  Returns - Euclidean distance between two arrays
  """
  return np.linalg.norm(np.array(a)- np.array(b), axis=ax)

fd = "Fourier Descriptor from final_FD() function"
N = len(fd)

pdist = np.zeros((N, N))

for i in range(len(fd)):
  for j in range(len(fd)):
    pdist[i,j] = np.linalg.norm(np.array(fd[i])- np.array(fd[j]))
    pdist[j,i]

clustering = DBSCAN(eps=0.5, min_samples=1, metric='precomputed')
clustering.fit(pdist)
labels = clustering.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = pdist[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = pdist[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()