import numpy as np, PointSet, time, copy

class Coreset(PointSet.PointSet):

    def __init__(self, P=None, W=None, _sense_bound_lambda=None, max_ellipsoid_iters=10, use_svd=True, problem_type=None):
        PointSet.PointSet.__init__(self, P=P, W=W, ellipsoid_max_iters=max_ellipsoid_iters, problem_type=problem_type, use_svd=use_svd)
        self.rank = 0
        self.sensitivities = None
        self._sense_bound_lambda = _sense_bound_lambda
        self.probability = None
        self.S = None
        self.rank = 0
        self.use_svd = use_svd
        self.time_taken = 0
        self.problem_type = problem_type

    def sampleCoreset(self, P, sensitivity, sample_size, random_state=0):
        startTime = time.time()
        weights = P.W
        t = np.sum(sensitivity)
        self.probability = sensitivity.flatten() / t
        n = P.n
        np.random.seed(random_state)
        indxs = np.random.choice(n, sample_size, p=(self.probability.flatten()))
        hist = np.histogram(indxs, bins=(range(n)))[0].flatten()
        indxs = copy.deepcopy(np.nonzero(hist)[0])
        
        # For some reason, the following line does not work in Python 3.8
        # S = P.P[(indxs, None[:None])]
        S = P.P[(indxs, slice(None))]
        
        weights = np.asarray((np.multiply(weights[indxs], hist[indxs])), dtype=float).flatten()
        weights = np.multiply(weights, 1.0 / (self.probability[indxs] * sample_size))
        self.time_taken = time.time() - startTime
        self.S = PointSet.PointSet(P=S, W=weights, ellipsoid_max_iters=(self.ellipsoid_max_iters), problem_type=(self.problem_type),
          use_svd=(self.use_svd),
          compute_U=False)
        return (self.S, self.time_taken)

    def computeSensitivity(self, P):
        sensitivitiy = np.empty(P.n)
        sensitivity = np.empty((P.n,))
        if "lz" in self.problem_type:
            sensitivity = self._sense_bound_lambda(P.P, P.W, P.d)
        else:
            sensitivity[P.pos_idxs] = self._sense_bound_lambda(x=(P.U[(P.pos_idxs, slice(None))]), w=(P.W[P.pos_idxs]), args=(
             P.sum_weights_pos, P.sum_weights_neg,
             P.sum_weights))
            sensitivity[P.neg_idxs] = self._sense_bound_lambda(x=(P.U[(P.neg_idxs, slice(None))]), w=(P.W[P.neg_idxs]), args=(
             P.sum_weights_neg, P.sum_weights_pos,
             P.sum_weights))
        return sensitivity

    def mergeCoresets(self, coreset, sample_size):
        self.S.mergePointSet(coreset.S)
        sens = self.computeSensitivity(P=(self.S))
        self.sampleCoreset(P=(self.S), sensitivity=sens, sample_size=sample_size)
        self.rank += 1
