from sklearn.manifold import LocallyLinearEmbedding as LLE  # for LLE dimensionality reduction


def run_lle(num_neighbors, dims, mthd, data):
    # Specify LLE parameters
    embed_lle = LLE(n_neighbors=num_neighbors,  # default=5, number of neighbors to consider for each point.
                    n_components=dims,  # default=2, number of dimensions of the new space
                    reg=0.001,
                    # default=1e-3, regularization constant, multiplies the trace of the local covariance matrix of the distances.
                    eigen_solver='auto',
                    # {‘auto’, ‘arpack’, ‘dense’}, default=’auto’, auto : algorithm will attempt to choose the best method for input data
                    # tol=1e-06, # default=1e-6, Tolerance for ‘arpack’ method. Not used if eigen_solver==’dense’.
                    # max_iter=100, # default=100, maximum number of iterations for the arpack solver. Not used if eigen_solver==’dense’.
                    method=mthd,  # {‘standard’, ‘hessian’, ‘modified’, ‘ltsa’}, default=’standard’
                    # hessian_tol=0.0001, # default=1e-4, Tolerance for Hessian eigenmapping method. Only used if method == 'hessian'
                    modified_tol=1e-12,
                    # default=1e-12, Tolerance for modified LLE method. Only used if method == 'modified'
                    neighbors_algorithm='auto',
                    # {‘auto’, ‘brute’, ‘kd_tree’, ‘ball_tree’}, default=’auto’, algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance
                    random_state=42,
                    # default=None, Determines the random number generator when eigen_solver == ‘arpack’. Pass an int for reproducible results across multiple function calls.
                    n_jobs=-1  # default=None, The number of parallel jobs to run. -1 means using all processors.
                    )
    # Fit and transofrm the data
    result = embed_lle.fit_transform(data)

    # Return results
    return result
