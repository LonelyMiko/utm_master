from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from py.scripts.plot import plot3D, plot2D_PX
from py.scripts.LLE_helper import run_lle
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

# Part 1
# 1. Generate the swis-roll dataset with 2000 points using the function datasets.make_swiss_roll
swiss_roll, y = make_swiss_roll(2000)

# 2. Apply the PCA and plot the data.
pca = PCA()
swiss_roll_pca = pca.fit_transform(swiss_roll)
plot3D(swiss_roll_pca, y, "PCA Swiss Roll")

# 3. Apply LLE (Local Linear Embedding) with 5 neighbours (manifold.locally_linear_embedding) by printing the error.
# Change the number of neighbours from 2 to 15 and plot the error line. Which is the best number of neighbours?
try:
    std_lle_res = run_lle(num_neighbors=2, dims=2, mthd='standard', data=swiss_roll_pca)
    plot2D_PX(std_lle_res, y, "Regular Swiss Roll - LLE")
except ValueError as e:
    print("When we run LLE with 2 neighbors, we get an error: ")
    print(e)

std_lle_res = run_lle(num_neighbors=15, dims=2, mthd='standard', data=swiss_roll)
plot2D_PX(std_lle_res, y, "Regular Swiss Roll - LLE")
# Which is the best number of neighbours?
# neighbors is often decided by the distances among samples. Especially, if you know the classes of your samples,
# you'd better set neighbors a little greater than the number of samples in each class.

# 4. Use Multi Dimensional Scaling with manifold.MDS and visualize the dataset in 2 dimension.
md_scaling = MDS().fit_transform(X=swiss_roll)
plot2D_PX(md_scaling, y, "MDS")

# 5. Apply t-SNE model to the same dataset with manifold.TSNE.Visualize the dataset
tsne_scaling = TSNE().fit_transform(X= swiss_roll)
plot2D_PX(tsne_scaling, y, "T-distributed Stochastic Neighbor Embedding").show()