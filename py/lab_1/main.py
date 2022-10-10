from sklearn import datasets, preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import plotly.tools as tls
import plotly.offline as py
import numpy
from sklearn.manifold import TSNE
from py.scripts.plot import plot2D

colors = ["navy", "turquoise", "darkorange"]
iris = datasets.load_iris()
y = iris.target
target_names = iris.target_names

if __name__ == '__main__':
    # Chapter A
    # print(iris)

    # Chapter B and C
    # Create the following matrix:
    matrix = numpy.array([
        [1, -1, 2],
        [2, 0, 0],
        [0, 1, -1]
    ])

    # Create the following matrix:
    second_matrix = numpy.array([
        [1, -1, 2],
        [2, 0, 0],
        [0, 1, -1]
    ])

    # Print X and compute the mean and the variance of X
    scaler = preprocessing.StandardScaler().fit(matrix)

    min_max_scale = MinMaxScaler()
    min_max_scale.fit(second_matrix)

    print("Matrix: " + str(matrix))
    print("Matrix Mean: " + str(scaler.mean_))
    print("Scale: " + str(scaler.scale_))

    # Normalize the data using MinMaxScaler.Print the scaled matrix and compute the mean and the variance.
    print("Second Matrix: " + str(matrix))
    print("Second Matrix Mean: " + str(second_matrix.mean()))
    print("Second Scale: " + str(min_max_scale.scale_))

    # Chapter D
    # Plot the data points into 2D dimension
    title = "I don't know what I'm doing, but it's work"
    plot2D(title, colors, target_names, iris.data, y).show()


    # Compute the correlations by using the corrcoef function
    corrcoef = numpy.corrcoef(iris.data)

    #Subplots in matplotlib
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot([1, 2, 3, 4, 5], [10, 5, 10, 5, 10], 'r-')
    ax2 = fig.add_subplot(222)
    ax2.plot([1, 2, 3, 4], [1, 4, 9, 16], 'k-')
    ax3 = fig.add_subplot(223)
    ax3.plot([1, 2, 3, 4], [1, 10, 100, 1000], 'b-')
    ax4 = fig.add_subplot(224)
    ax4.plot([1, 2, 3, 4], [0, 0, 1, 1], 'g-')
    plt.tight_layout()
    fig = plt.gcf()
    plotly_fig = tls.mpl_to_plotly(fig)
    plotly_fig['layout']['title'] = 'Simple Subplot Example'
    plotly_fig['layout']['margin'].update({'t': 40})
    py.plot(plotly_fig)

    # Chapter E
    # 1. Use the correlations information’s found in D.3 and reduce the dataset to 3 variables then to 2 variables.
    X = corrcoef ** 2
    X_mean = numpy.mean(X)
    y = iris.target
    target_names = iris.target_names

    # Analyze the help of these functions (pca and lda) and apply them on the Iris dataset
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LDA(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each component
    print(
        "explained variance ratio PCA (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    print(
        "explained variance ratio LDA (first two components): %s"
        % str(lda.explained_variance_ratio_)
    )

    print("Corrcoef mean: " + str(X_mean))

    title = "PCA of IRIS dataset"
    plot2D(title, colors, target_names, X_r, y).show()

    title = "LDA of IRIS dataset"
    plot2D(title, colors, target_names, X_r2, y).show()

    # Use another dimensional reduction technique
    tsne = TSNE(random_state=0)
    X_normalized = preprocessing.StandardScaler().fit(iris.data).transform(iris.data)
    X_tsne = tsne.fit_transform(X_normalized)

    title = "X_tsne of IRIS dataset"
    plot2D(title, colors, target_names, X_tsne, y).show()