import numpy as np
import pandas as pd

from scipy import stats
from scipy.special import gamma
from scipy.spatial import KDTree

from numba import njit, jit

# TO EXPORT: method_silverman, method_abramson, method_RVKDE, method_ERVKDE, MSE

@njit
def kde(x, mu, sigma, DIMENSION=2):
    """
    f_hat(v) -> density at x
    
    For: generate_abramson, get_density
    """
    dist_sq = np.sum((x - mu)**2, axis=1)
    kde_val = (1/((sigma**2)*2*np.pi))**(0.5*DIMENSION)*np.exp(-dist_sq/(2*(sigma**2)))
    return np.mean(kde_val)


def get_density(xs, ys, mu, sigma, DIMENSION=2):
    """
    xs, ys -> kde(x,y) for x in xs, y in ys

    Dependencies: kde
    For: abramson_sigmas, main
    """
    return np.array([[kde(np.array([x,y]), mu, sigma, DIMENSION) for x in xs] for y in ys])


@njit
def ISE_loop(mu, s, DIMENSION=2):
    """
    Optimized loop for abramson_sigmas()
    
    mu (np.array)
    s: sigma (one value)
    """
    total = 0
    for i in range(len(mu)):
        for j in range(len(mu)):
            dist_sq = np.sum((mu[i]-mu[j])**2)
            total += (i != j)*(1/(s*s*2*np.pi))**(0.5*DIMENSION)*np.exp(-dist_sq/(2*s*s))
    return (2*total/len(mu)/(len(mu)-1))


@jit(looplift=True)
def abramson_sigmas(samples, xs, ys, DIMENSION=2):
    """
    samples, xs, ys -> samples (mu), abramson sigmas
    
    Dependencies: kde
    """
    xy_interval = abs(xs[1] - xs[0])*abs(ys[1] - ys[0])

    sigma_range = [2**x for x in range(-4, 4)]
    best_s, best_ISE_est = 0, np.inf
    N = len(samples)
    
    # Estimate the best sigma value
    for s in sigma_range:
        # Compute Fixed Gaussian KDE
        sigma = np.array([s] * N)
        density_squared_sum = np.square(get_density(xs, ys, samples, sigma)).sum()

        # Get ISE estimate
        ISE_estimate = -1 * ISE_loop(samples, s, DIMENSION) + density_squared_sum * xy_interval
        
        if ISE_estimate < best_ISE_est:
            best_s, best_ISE_est = s, ISE_estimate

    fixed_sigmas = np.array([best_s] * N)
    
    # Use estimate to compute sigmas
    sigmas = []
    for center in samples:
        sigmas.append(1 / np.sqrt(kde(center, samples, fixed_sigmas)))
        
    sigmas = np.array(sigmas)
    
    # Make similar to fixed gaussian
    sigmas = sigmas / sigmas.mean() * best_s
    
    return samples, sigmas


@jit(looplift=True)
def rvkde_sigmas(samples, K=None, BETA=2, DIMENSION=2, smoothing=True):
    """
    samples -> rvkde sigmas
    
    K: if None, use defaults
    
    For: method_RVKDE, method_ERVKDE
    """
    if K==None:
        K = int(len(samples)/10)
    
    tree = KDTree(samples, int(K*1.5))
    
    R_scale = np.sqrt(np.pi)/np.power((K+1)*gamma(DIMENSION/2+1), 1./DIMENSION)
    sigma_scale = R_scale*BETA*(DIMENSION+1.)/DIMENSION/K
    
    # Calculate sigmas
    sigmas = []
    for x in samples:
        # Find K nearest neighbor
        knn = tree.query(x, K+1)
        sigma = np.sum(knn[0])*sigma_scale
        sigmas.append(sigma)
    
    # Sigma smoothing
    new_sigmas = []
    if smoothing:
        for x in samples:
            neighbor_indices = tree.query(x, K+1)[1][1:K+1] # indices of nearest-neighbors
            new_sigma = np.sum([sigmas[i] for i in neighbor_indices])/K
            new_sigmas.append(new_sigma)
    else:
        new_sigmas = sigmas
            
    new_sigmas = np.array(new_sigmas)

    return samples, new_sigmas


def method_silverman(data, tofit):
    """
    data, tofit (X,Y axes) -> Silverman KDE density
    
    data: numpy array of 2D samples
    tofit: [x,y] pairs for X, Y np.linspace axes
    
    Example tofit:
       tofit = np.array([[x,y] for y in ys for x in xs])
          ( where xs = ys = np.linspace(-4,4,100) )
    
    Dependencies: scipy.stats
    """
    gaus_kde = stats.gaussian_kde(data.T)
    return gaus_kde(tofit.T)


def method_abramson(data, xs, ys, DIMENSION = 2):
    """
    data, xs, ys -> Abramson KDE density
    
    data: numpy array of 2D samples
    xs, ys: np.linspace(start, end, num_points) for X, Y axes
       (represents X and Y axes)
    """
    mu, sigma = abramson_sigmas(data, xs, ys, DIMENSION)
    return get_density(xs, ys, mu, sigma, DIMENSION)


def method_RVKDE(data, xs, ys, DIMENSION = 2):
    """
    data, xs, ys -> RVKDE density
    
    data: numpy array of 2D samples
    xs, ys: np.linspace(start, end, num_points) for X, Y axes
       (represents X and Y axes)
    """
    mu, sigma = rvkde_sigmas(data, int(len(data)/10), DIMENSION)
    return get_density(xs, ys, mu, sigma, DIMENSION)


def method_ERVKDE(data, xs, ys, DIMENSION = 2):
    """
    data, xs, ys -> ERVKDE density
    
    data: numpy array of 2D samples
    xs, ys: np.linspace(start, end, num_points) for X, Y axes
       (represents X and Y axes)
    """
    mu, sigma = rvkde_sigmas(data, int(len(data)/10), DIMENSION)
    sig_avg = np.mean(np.std(data))
    diff = ((4*sig_avg**5/(len(data)*(DIMENSION+2)))**(1/(DIMENSION+4))) - np.median(sigma)
    elevated_sigma = np.array([s + diff for s in sigma])
    return get_density(xs, ys, mu, elevated_sigma, DIMENSION)


def MSE(predicted_density, true_density):
    """Mean Square Error"""
    predicted_density = np.array(predicted_density)
    true_density = np.array(true_density)
    mse = ((predicted_density-true_density)**2).mean()
    return mse

