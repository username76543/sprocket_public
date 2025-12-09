from numba import get_num_threads, njit, prange, set_num_threads
import numba as nb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial.distance import chebyshev, correlation

@nb.njit(inline='always')
def linear_penalty(leftover_dirt, sliced_series_1, sliced_series_2, weight=.5):
    return abs(leftover_dirt * weight)    

@nb.njit(
    parallel=True,
    fastmath=True,
    cache=True,
)
def compute_signed_emd(series_1, series_2):#, penalty_fn, penalty_args):
    """
    Computes a Signed EMD: EMD(Positive components) + EMD(|Negative components|).
    
    Inputs series_1 and series_2 are expected to be 2D arrays (n_channels, n_timepoints).
    Penalty for unbalanced mass is .5, minimum to be a metric
    """
    n_channels, n_timepoints = series_1.shape
    
    total_emd = 0.0
    
    for i in nb.prange(n_channels):
        s1 = series_1[i]
        s2 = series_2[i]
        
        leftover_dirt_pos = 0.0
        total_dirt_pos = 0.0
        leftover_dirt_neg = 0.0
        total_dirt_neg = 0.0
        
        for j in range(n_timepoints):
            val1 = s1[j]
            val2 = s2[j]

            s1_pos_j = max(val1, 0.0)
            s2_pos_j = max(val2, 0.0)
            
            s1_neg_j = max(-val1, 0.0)
            s2_neg_j = max(-val2, 0.0)
            
            leftover_dirt_pos = s1_pos_j + leftover_dirt_pos - s2_pos_j
            total_dirt_pos += abs(leftover_dirt_pos)
            
            leftover_dirt_neg = s1_neg_j + leftover_dirt_neg - s2_neg_j
            total_dirt_neg += abs(leftover_dirt_neg)
        
        
        total_emd += total_dirt_pos + total_dirt_neg + .5*(leftover_dirt_pos+leftover_dirt_neg)#penalty_fn(leftover_dirt_pos+leftover_dirt_neg, s1, s2, *penalty_args)
        
    return total_emd
    
@nb.njit(
    fastmath=True,
    cache=True,
)
def sorted_emd_wrapper(series_1, series_2):#, index_tuples, penalty_fn, penalty_args):
    """
    Wrapper for compute_emd_with_nonuniform_penalty that sorts each channel independently
    before computing the signed EMD.
    
    Inputs:
        series_1, series_2 : 2D arrays (n_channels, n_timepoints)
        penalty_fn : Numba-jitted penalty function
        penalty_args : tuple of additional arguments for penalty_fn
        
    Returns:
        total_emd : float
    """
    n_channels, n_timepoints = series_1.shape
    
    # Allocate arrays for sorted series
    sorted_s1 = np.empty_like(series_1)
    sorted_s2 = np.empty_like(series_2)
    
    # Sort each channel independently
    for i in range(n_channels):
        sorted_s1[i] = np.sort(series_1[i])
        sorted_s2[i] = np.sort(series_2[i])
    
    # Call the original compute_signed_emd
    total_emd = compute_signed_emd(
        sorted_s1,
        sorted_s2)
    
    return total_emd

@nb.njit(fastmath=True, cache=True)
def cosine_similarity_numba(u, v):
    #assert(u.shape == v.shape)
    u_len = u.shape[0]
    uv = 0
    uu = 0
    vv = 0
    for i in range(u_len):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu * vv)
    return cos_theta

@nb.njit(
    fastmath=True,
    cache=True,
)
def cosine_similarity_wrapper(series_1, series_2):#, index_tuples, penalty_fn, penalty_args):
    """
    Wrapper for sklearn cosine similarity
    
    Inputs:
        series_1, series_2 : 2D arrays (n_channels, n_timepoints)
        
    Returns:
        total_dist : float
    """
    total_dist = 0.0
    
    n_channels = series_1.shape[0]
    
    for i in range(n_channels):
        total_dist += cosine_similarity_numba(series_1[i], series_2[i])
    
    return total_dist
    
@nb.njit(fastmath=True, cache=True, inline='always')
def chebyshev_distance(x, y):
    """Compute the Chebyshev distance between two points."""
    return np.max(np.abs(x - y))

@nb.njit(
    fastmath=True,
    cache=True,
)
def chebyshev_wrapper(series_1, series_2):#, index_tuples, penalty_fn, penalty_args):
    """
    Wrapper for chebyshev from scipy
    
    Inputs:
        series_1, series_2 : 2D arrays (n_channels, n_timepoints)
        
    Returns:
        total_dist : float
    """
    total_dist = 0.0
    
    n_channels = series_1.shape[0]
    
    # Sort each channel independently
    for i in range(n_channels):
        total_dist += chebyshev_distance(series_1[i], series_2[i])
    
    return total_dist
    
@nb.njit(fastmath=True, cache=True)
def correlation_distance(x, y):
    # Compute means
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    
    # Center the vectors
    shifted_x = x - mu_x
    shifted_y = y - mu_y
    
    # Compute dot product and norms
    dot_product = np.dot(shifted_x, shifted_y)
    norm_x = np.sqrt(np.dot(shifted_x, shifted_x))
    norm_y = np.sqrt(np.dot(shifted_y, shifted_y))
    
    # Handle edge cases
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0  # or 1.0 depending on desired behavior
    
    # Compute correlation and distance
    correlation = dot_product / (norm_x * norm_y)
    return 1.0 - correlation

@nb.njit(
    fastmath=True,
    cache=True,
)
def correlation_wrapper(series_1, series_2):#, index_tuples, penalty_fn, penalty_args):
    """
    Wrapper for correlation distance from scipy
    
    Inputs:
        series_1, series_2 : 2D arrays (n_channels, n_timepoints)
        
    Returns:
        total_dist : float
    """
    total_dist = 0.0
    
    n_channels = series_1.shape[0]
    
    # Sort each channel independently
    for i in range(n_channels):
        total_dist += correlation_distance(series_1[i], series_2[i])
    
    return total_dist
