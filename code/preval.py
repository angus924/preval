# Angus Dempster, Geoffrey I Webb, Daniel F Schmidt

# Prevalidated ridge regression is a highly-efficient drop-in replacement
# for logistic regression for high-dimensional data

# https://arxiv.org/abs/

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import LabelBinarizer

# == constants =================================================================

EPS = np.finfo(np.float32).eps
LOG_EPS = np.log(EPS)

# == softmax function ==========================================================

def _softmax(X):

    exp_X = np.exp(X.clip(LOG_EPS, -LOG_EPS))

    return exp_X / np.sum(exp_X, axis = -1, keepdims = True)

# == log loss function for scipy.optimize.minimize =============================

def _log_loss(c, *args):

    Y, Y_loocv, B0 = args

    P = _softmax(c * Y_loocv + B0)

    return -np.log((Y * P).max(1)).sum()

# == PreVal ====================================================================

class PreVal():

    def __init__(self, lambdas = np.logspace(-3, 3, 10)):

        self.lambdas = lambdas.astype(np.float32)

        self._is_fitted = False

    def fit(self, X, Y):

        X = X.astype(np.float32, copy = False)
        
        # drop low-variance columns
        self._mask = X.std(0) < 1e-6
        X = X[:, ~self._mask]

        X = np.hstack((np.ones((X.shape[0], 1), dtype = np.float32), X))

        n, p = X.shape

        # encode class as regression target, Y in {-1, +1}
        self._lb = LabelBinarizer(neg_label = -1)
        Y = self._lb.fit_transform(Y).astype(np.float32)

        # fix for binary classes
        if Y.shape[-1] == 1:
            Y = np.hstack((-Y, Y))

        # cetnre Y
        self.B0 = Y.mean(0)
        Y = Y - self.B0

        # svd via eigendecomposition
        # on X^T X (for n >= p)
        # on X X^T (for n <  p)
        if n >= p:

            batch_size = 4_096

            G = np.zeros((p, p), dtype = np.float32)
            for i in range(0, X.shape[0], batch_size):
                G = G + (X[i:i + batch_size].T @ X[i:i + batch_size])
            S2, V = np.linalg.eigh(G)
            S2 = S2.clip(EPS)
            S = np.sqrt(S2)
            U = (X @ V) * (1 / S)
        
        else:

            G = X @ X.T
            S2, U = np.linalg.eigh(G)
            S2 = S2.clip(EPS)
            S = np.sqrt(S2)
            V = (X.T @ U) * (1 / S)

        R = U * S
        R2 = R ** 2
        RTY = R.T @ Y # "Q" in paper

        best_loss = np.inf
        self.c = np.float32(1.0)
        self.lambda_ = None

        for lambda_ in self.lambdas:

            alpha_hat = (1 / (S2[:, None] + lambda_)) * RTY
            
            Y_hat = R @ alpha_hat

            # "full fit" residuals for given alpha
            E = Y - Y_hat

            # diagonal of hat matrix
            diag_H = (R2 / (S2 + lambda_)).sum(1)
            
            # loocv residuals
            E_loocv = E / (1 - diag_H[:, None]).clip(EPS)

            # difference between overall residuals and loocv residuals
            delta = E_loocv - E

            # loocv predictions
            Y_loocv = Y_hat - delta

            result = \
            minimize(
                fun     = _log_loss,
                x0      = 1.0,
                args    = (Y, Y_loocv, self.B0),
                method  = "BFGS",
                jac     = "2-point",
            )
            # use of Y_hat in place of Y_loocv in minimize gives "naive scaling"

            nll = result.fun

            if nll < best_loss:
                best_loss = nll
                self.c = np.float32(result.x.item())
                self.lambda_ = lambda_
                alpha_hat_best = alpha_hat

        self.B = self.c * (V @ alpha_hat_best)

        self._is_fitted = True

    def predict_proba(self, X):

        assert self._is_fitted, "not fitted"

        X = X.astype(np.float32, copy = False)

        X = X[:, ~self._mask]

        X = np.hstack((np.ones((X.shape[0], 1), dtype = np.float32), X))

        return _softmax(X @ self.B + self.B0)

    def predict(self, X):

        assert self._is_fitted, "not fitted"

        return self._lb.classes_[self.predict_proba(X).argmax(1)]