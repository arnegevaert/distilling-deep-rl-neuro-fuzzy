import numpy as np
from sklearn import mixture


def subtractive_clustering(X, ra=0.1, rb=0.15, eps_up=0.4, eps_down=0.15):
    # CALCULATE CLUSTER CENTERS
    # Initialize potentials
    center_indices = []
    centers = []
    potentials = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        potentials += np.exp(-(4/ra**2) * np.sum((X - X[i])**2, axis=1))

    # First cluster center
    c_index = np.argmax(potentials)
    center_indices.append(c_index)
    centers.append(X[c_index])
    first_potential = potentials[c_index]

    done = False
    while not done:
        # Get max potential and reduce potentials
        pmax = potentials[c_index]
        xmax = X[c_index]
        for i in range(X.shape[0]):
            potentials[i] -= pmax * np.exp(-(4/rb**2) * np.sum((X[i]-xmax)**2))
        c_index = np.argmax(potentials)
        if potentials[c_index] > eps_up * first_potential:
            # If large enough potential, add to clusters
            centers.append(X[c_index])
            center_indices.append(c_index)
        elif potentials[c_index] < eps_down * first_potential:
            # If too small potential, end algorithm
            done = True
        else:
            # Edge case (see dourado paper)
            dmin = min([np.sum((X[c_index] - X[other_index])**2) for other_index in center_indices])
            if dmin/ra + potentials[c_index]/first_potential >= 1:
                centers.append(X[c_index])
                center_indices.append(c_index)
            else:
                potentials[c_index] = 0
                c_index = np.argmax(potentials)
    centers = np.vstack(centers)

    # CALCULATE PER-DIMENSION SAMPLE VARIANCE
    distances = [[[] for _ in range(X.shape[1])] for _ in range(centers.shape[0])]
    for i in range(X.shape[0]):
        # For each input, calc distance to corresponding input of each center state
        center_distances = np.abs(X[i, :] - centers)
        # For each input, get center state that is closest
        min_distances = np.min(center_distances, axis=0)
        for j in range(X.shape[1]):
            closest = int(np.where(center_distances[:, j] == min_distances[j])[0][0])
            # Record distance to corresponding input of closest center state
            distances[closest][j].append(min_distances[j])

    # Avg of distance to each input of each center is variance
    sigma = np.ones((centers.shape[0], X.shape[1])) * 1e-5
    for i in range(centers.shape[0]):
        for j in range(X.shape[1]):
            d = distances[i][j]
            if len(d) > 0:
                sigma[i, j] = np.average(d)
    return centers, sigma


def gmm_clustering(X, n_components):
    gmm = mixture.GaussianMixture(n_components, covariance_type="diag").fit(X)
    return gmm.means_, gmm.covariances_

