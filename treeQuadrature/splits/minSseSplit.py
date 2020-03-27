import numpy as np

def minSseSplit(container, **kwargs):
    '''Partition into k sets that minimize variance of f over each set'''

    samples = container.X
    dims = samples.shape[1]

    ys = container.y.reshape(-1)
    
    best_dimension = -1
    best_thresh = np.inf
    best_score = np.inf
    
    # Find best split for each dimension and take best
    for dim in range(dims):
        xs = np.array(samples[:, dim], copy=True)
        indices = np.argsort(xs)
        xss = xs[indices]
        yss = np.array(ys[indices], copy=True)
        
        thresh, score = findMinSplit(xss, yss)
        
        if score < best_score:
            best_dimension = dim
            best_thresh = thresh
            best_score = score

    lcont, rcont = container.split(best_dimension, best_thresh)

    return [lcont, rcont]


def findMinSplit(xs, ys):
        '''Partition xs and ys such that variance across ys subsets is minimized'''
        # xs, ys both 1D array
        # xs is sorted, ys aligned with xs
        
        best_thresh = np.inf
        best_score = np.inf
        
        threshes = []
        scores = []
        
        # Iterate through all possible splits :(
        for i in range(1, ys.shape[0]):
            lvar = np.var(ys[:i])
            rvar = np.var(ys[i:])
            
            thresh = xs[i-1]

            # Compute SSE Impurity
            n = ys.shape[0]
            nl = i
            nr = n - i
            score = (nl*lvar) + (nr*rvar)
            
            threshes.append(thresh)
            scores.append(score)
            
            if score < best_score:
                best_thresh = thresh
                best_score = score
        
        return best_thresh, best_score
