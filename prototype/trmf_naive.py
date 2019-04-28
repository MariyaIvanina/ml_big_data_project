import numpy as np


class TRMFNaive:
    """Simple basic implementation of Temporal Regularized Matrix Factorization.
    Parameters
    ----------

    lags : array-like, shape (n_lags,)
        Set of lag indices to use in model.

    rank : int
        Length of latent embedding dimension

    lambda_f : float
        Regularization parameter used for matrix F.

    lambda_x : float
        Regularization parameter used for matrix X.

    lambda_w : float
        Regularization parameter used for matrix W.
    alpha : float
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.

    eta : float
        Regularization parameter used for X when undercovering autoregressive dependencies.
    num_iter : int
        Number of iterations of updating matrices F, X and W.
    f_step : float
        Step of gradient descent when updating matrix F.
    x_step : float
        Step of gradient descent when updating matrix X.
    w_step : float
        Step of gradient descent when updating matrix W.

    Attributes
    ----------
    F : ndarray, shape (n_timeseries, rank)
        Latent embedding of timeseries.
    X : ndarray, shape (rank, n_timepoints)
        Latent embedding of timepoints.
    W : ndarray, shape (rank, n_lags)
        Matrix of autoregressive coefficients.
    """

    def __init__(self, lags, rank, lambda_f, lambda_x, lambda_w, alpha, eta, num_iter=1000,
                 f_step=0.0001, x_step=0.0001, w_step=0.0001):
        self.lags = lags
        self.L = len(lags)
        self.rank = rank
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.alpha = alpha
        self.eta = eta
        self.num_iter = num_iter
        self.f_step = f_step
        self.x_step = x_step
        self.w_step = w_step

    def fit(self, train_data):
        """Fits TRMF model according to the given training data.
        Each matrix is updated with gradient descent.

        Parameters
        ----------
        train_data : ndarray, shape (n_timeseries, n_timepoints)
            Training data.
        """
        self.Y = train_data
        mask = np.array((~np.isnan(self.Y)).astype(int))
        self.mask = mask
        self.Y[self.mask == 0] = 0.

        self.N, self.T = self.Y.shape
        self.W = np.random.randn(self.rank, self.L) / self.L
        self.F = np.random.randn(self.N, self.rank)
        self.X = np.random.randn(self.rank, self.T)

        self._optimize()

    def predict(self, horizon):
        """Predicts each of timeseries 'horizon' timepoints ahead.

        Parameters
        ----------
        horizon : int
            Number of timepoints to forecast.
        Returns
        -------
        preds : ndarray, shape (n_timeseries, T)
            Predictions.
        """
        zeros = np.zeros((self.rank, horizon))
        X_extended = np.hstack([self.X, zeros])
        for t in range(self.T, self.T + horizon):
            for l in range(self.L):
                lag = self.lags[l]
                X_extended[:, t] += X_extended[:, t - lag] * self.W[:, l]
        X_predicted = X_extended[:, self.T:]
        return np.dot(self.F, X_predicted)

    def _optimize(self):
        for _ in range(self.num_iter):
            self.F -= self.f_step * self._grad_f()
            self.X -= self.x_step * self._grad_x()
            self.W -= self.w_step * self._grad_w()

    def _grad_f(self):
        return - 2 * np.dot((self.Y - np.dot(self.F, self.X)) * self.mask, self.X.T) \
               + 2 * self.lambda_f * self.F

    def _grad_x(self):
        z_1 = z_2 = None
        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.rank, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (np.roll(self.X, -lag, axis=1) - X_l) * W_l
            z_2[:, -lag:] = 0.
        grad_T_x = z_1 + z_2

        return - 2 * np.dot(self.F.T, self.mask * (self.Y - np.dot(self.F, self.X))) \
               + self.lambda_x * grad_T_x + self.eta * self.X

    def _grad_w(self):
        grad = np.zeros((self.rank, self.L))
        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.rank, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (z_1 * np.roll(self.X, lag, axis=1)).sum(axis=1)
            grad[:, l] = z_2
        return grad + self.W * 2 * self.lambda_w / self.lambda_x \
               - self.alpha * 2 * (1 - self.W.sum(axis=1)).repeat(self.L).reshape(self.W.shape)
