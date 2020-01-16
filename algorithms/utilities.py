# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 09:34:00 2018

@author: Hassen Dhrif
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import cross_val_score
from typing import List


def subset_accuracy(p, sw) -> float:
    """

    :param p:
    :param sw:
    :return:
    """

    subset = sw.X[:, p.b==1]

    f1 = cross_val_score(sw.clf, subset, sw.y, cv=sw.cv, scoring='recall', n_jobs=-1).mean()
    f2 = sum(sw.irdc[p.b==1])/p._nbf
    max_freq = max(sw.freq)
    if max_freq > 1:
        f3 = sum([(f-1)/(max_freq-1) for f in sw.freq[p.b==1]])/p._nbf
    else:
        f3 = 0
    f = sw.alpha_1*f1 + sw.alpha_2*f2 + sw.alpha_3*f3
    return 1-f, f1


def sigmoid(x):
    """
    calculate sigmoid function for binary feature selection pso
    :param x:
    :return:
    """
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    """

    :param x:
    :return:
    """
    s = [1 if i > 0 else 0 for i in x]
    return np.array(s)


def RDC(x, y, f=np.sin, k=20, s=1/6., n=10):
    """
    Implements the Randomized Dependence Coefficient
    David Lopez-Paz, Philipp Hennig, Bernhard Schoelkopf
    http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf

    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and return the median (for stability)

    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(RDC(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError:
                pass
        return np.median(values)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T]) / float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T]) / float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s / X.shape[1]) * np.random.randn(X.shape[1], k)
    Ry = (s / Y.shape[1]) * np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0 + k, k0:k0 + k]
        Cxy = C[:k, k0:k0 + k]
        Cyx = C[k0:k0 + k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.inv(Cxx), Cxy),
                                        np.dot(np.linalg.inv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))