import numpy as np
from sklearn.neighbors import NearestNeighbors


def normalize(x, return_scalers=False):

    def _normalize_cloud(x):

        x_mean = np.mean(x, axis=0)
        x_norm = np.copy(x - x_mean)
        x_max = np.max(np.sqrt(np.sum(np.square(x_norm), axis=1)))
        x_norm = x_norm / x_max

        return x_norm, x_mean, x_max

    n_clouds, n_points, n_dims = x.shape
    x_norm = np.zeros([n_clouds, n_points, n_dims])

    x_mean = np.zeros([n_clouds, n_dims])
    x_max = np.zeros([n_clouds, 1])

    fid_lst = range(0, n_clouds)
    for pid in fid_lst:
        x_norm[pid], x_mean[pid], x_max[pid] = _normalize_cloud(x[pid])

    if return_scalers:
        return x_norm, x_mean, x_max
    else:
        return x_norm


def generate_random_basis_square(n_points=1000, n_dims=2, random_seed=13):

    np.random.seed(random_seed)
    x = np.random.uniform(low=-1.0, high=1.0, size=[n_points, n_dims])
    np.random.seed(None)

    return x

def generate_random_basis(n_points=1000, n_dims=2, radius=1.0, random_seed=13):
    """Sample uniformly from d-dimensional unit ball
    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html
    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection
    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """
    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms

    # now sample radiuses uniformly
    r = np.random.uniform(size=[n_points, 1])
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u
    np.random.seed(None)

    return x


def encode(x, n_bps_points=512, radius=1.5, random_seed=13, custom_basis=None):

    n_clouds, n_points, n_dims = x.shape
    if custom_basis is None:
        basis_set = generate_random_basis(n_bps_points, n_dims=n_dims, radius=radius, random_seed=random_seed)
    else:
        basis_set = custom_basis

    n_bps_points = basis_set.shape[0]

    x_bps = np.zeros([n_clouds, n_bps_points])
    fid_lst = range(0, n_clouds)

    for fid in fid_lst:
        nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(x[fid])
        fid_dist, npts_ix = nbrs.kneighbors(basis_set)
        x_bps[fid] = fid_dist.squeeze()

        return x_bps, npts_ix