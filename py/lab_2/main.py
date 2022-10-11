from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding

from py.scripts.plot import plot3D, plot2D_PX, plot_embedding
from py.scripts.LLE_helper import run_lle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import (
    Isomap,
    MDS,
    SpectralEmbedding,
    TSNE, LocallyLinearEmbedding,
)
from sklearn.datasets import load_digits
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection
from time import time
import matplotlib.pyplot as plt
from sklearn import tree

'''

TO VISUALIZE THE DATA, PLEASE REMOVE THE COMMENT FROM THE LINES,
PLEASE DELETE ONE COMMENT EACH TIME YOU RUN IT TO OPTIMIZE THE WAITING TIME:
42
43
55
62
66
133
154
'''

# Part 1
# 1. Generate the swis-roll dataset with 2000 points using the function datasets.make_swiss_roll
swiss_roll, y = make_swiss_roll(2000)

# 2. Apply the PCA and plot the data.
pca = PCA()
swiss_roll_pca = pca.fit_transform(swiss_roll)
# plot3D(swiss_roll_pca, y, "PCA Swiss Roll").show()
# plot2D_PX(swiss_roll_pca, y, "PCA Swiss Roll").show()

# 3. Apply LLE (Local Linear Embedding) with 5 neighbours (manifold.locally_linear_embedding) by printing the error.
# Change the number of neighbours from 2 to 15 and plot the error line. Which is the best number of neighbours?
try:
    std_lle_res = run_lle(num_neighbors=2, dims=2, mthd='standard', data=swiss_roll_pca)
    plot2D_PX(std_lle_res, y, "Regular Swiss Roll - LLE")
except ValueError as e:
    print("When we run LLE with 2 neighbors, we get an error: ")
    print(e)

std_lle_res = run_lle(num_neighbors=15, dims=2, mthd='standard', data=swiss_roll)
# plot2D_PX(std_lle_res, y, "Regular Swiss Roll - LLE").show()
# Which is the best number of neighbours?
# neighbors is often decided by the distances among samples. Especially, if you know the classes of your samples,
# you'd better set neighbors a little greater than the number of samples in each class.

# 4. Use Multi Dimensional Scaling with manifold.MDS and visualize the dataset in 2 dimension.
md_scaling = MDS().fit_transform(X=swiss_roll)
# plot2D_PX(md_scaling, y, "MDS").show()

# 5. Apply t-SNE model to the same dataset with manifold.TSNE.Visualize the dataset
tsne_scaling = TSNE().fit_transform(X=swiss_roll)
# plot2D_PX(tsne_scaling, y, "T-distributed Stochastic Neighbor Embedding").show()

# Part 2

digits = load_digits(n_class=6)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
n_neighbors = 30

embeddings = {
    "Random projection embedding": SparseRandomProjection(
        n_components=2, random_state=42
    ),
    "Truncated SVD embedding": TruncatedSVD(n_components=2),
    "Linear Discriminant Analysis embedding": LinearDiscriminantAnalysis(
        n_components=2
    ),
    "Isomap embedding": Isomap(n_neighbors=n_neighbors, n_components=2),
    "Standard LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="standard"
    ),
    "Modified LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="modified"
    ),
    "Hessian LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="hessian"
    ),
    "LTSA LLE embedding": LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method="ltsa"
    ),
    "MDS embedding": MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2),
    "Random Trees embedding": make_pipeline(
        RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
        TruncatedSVD(n_components=2),
    ),
    "Spectral embedding": SpectralEmbedding(
        n_components=2, random_state=0, eigen_solver="arpack"
    ),
    "t-SNE embeedding": TSNE(
        n_components=2,
        n_iter=500,
        n_iter_without_progress=150,
        n_jobs=2,
        random_state=0,
    ),
    "NCA embedding": NeighborhoodComponentsAnalysis(
        n_components=2, init="pca", random_state=0
    ),
}

projections, timing = {}, {}
for name, transformer in embeddings.items():
    if name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
    else:
        data = X

    print(f"Computing {name}...")
    start_time = time()
    projections[name] = transformer.fit_transform(data, y)
    timing[name] = time() - start_time

for name in timing:
    title = f"{name} (time {timing[name]:.3f}s)"
    plt.title(title)
    plot_embedding(projections[name], title, digits, y)
# plt.show()

# Use  a  classification  model  (Decision  Tree  for  example)  on  all  the projections and compute the errors
projections, timing = {}, {}
clf = tree.DecisionTreeClassifier()
for name, transformer in embeddings.items():
    if name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
    else:
        data = X

    print(f"Computing {name}...")
    start_time = time()
    projections[name] = clf.fit(data, y)
    timing[name] = time() - start_time

for name in timing:
    title = f"{name} (time {timing[name]:.3f}s)"
    tree.plot_tree(clf)
    plt.title(title)
    # plt.show()