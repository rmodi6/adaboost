import argparse

import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def weak_learner(X, y, D):
    m, d = X.shape
    F_star, theta_star, j_star = float('inf'), 0, 0
    for j in range(d):
        sorted_indexes = X[:, j].argsort()
        xj = X[sorted_indexes, j]
        xj = np.append(xj, xj[-1] + 1)
        y_sorted = y[sorted_indexes]
        D_sorted = D[sorted_indexes]
        F = np.sum(D_sorted[np.where(y_sorted == 1)])
        if F < F_star:
            F_star, theta_star, j_star = F, xj[0] - 1, j
        for i in range(m):
            F = F - y_sorted[i] * D_sorted[i]
            if F < F_star and xj[i] != xj[i + 1]:
                F_star, theta_star, j_star = F, (xj[i] + xj[i + 1]) / 2, j
    return theta_star, j_star


def adaboost(X, y):
    m, d = X.shape
    D = np.array([1 / m for i in range(m)])
    T = 10
    weak_learners, errors, weights = [], [], []

    for t in range(T):
        weak_learners.append(weak_learner(X, y, D))
        theta, j = weak_learners[-1]
        preds = np.where((theta >= X[:, j]), 1, -1)
        error = np.sum(D[np.where(preds != y)])
        errors.append(error)
        w = 0.5 * math.log(1 / error - 1)
        weights.append(w)
        D = D * np.exp(-w * y * preds)
        D = D / sum(D)
    # print(errors)
    return weak_learners, weights


def compute_error(X, y, weak_learners, weights):
    preds = np.zeros(y.shape)
    for i, learner in enumerate(weak_learners):
        theta, j = learner
        preds = preds + weights[i] * np.where((theta >= X[:, j]), 1, -1)
    preds = np.where(preds < 0, -1, 1)

    return len(np.where(preds != y)[0]) / m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset_path', action='store', type=str, help='path to dataset')
    parser.add_argument('--mode', dest='mode', action='store', type=str, help='mode of algorithm', default='erm')

    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)
    df.head()

    y = df.iloc[:, -1].values
    y = np.where(y == 0, -1, 1)

    X = df.iloc[:, :-1].values
    m, d = X.shape

    if args.mode == 'erm':
        weak_learners, weights = adaboost(X, y)
        error = compute_error(X, y, weak_learners, weights)
        print('Decision Stumps: %s \nWeights: %s \nError: %f' % (weak_learners, weights, error))
    elif args.mode == 'cv':
        m, d = X.shape
        k = 10
        s = int(m / k) + (1 if m % k != 0 else 0)
        batches = []

        indexes = list(range(X.shape[0]))
        random.shuffle(indexes)

        X = X[indexes]
        y = y[indexes]

        for i in range(k):
            start_index, end_index = s * i, s * (i + 1)
            batches.append((X[start_index:end_index], y[start_index:end_index]))

        errors = []
        for i in range(k):
            print('Executing Fold #: %d' % (i + 1))
            train_X, train_y, test_X, test_y = None, None, None, None
            for j, (X, y) in enumerate(batches):
                if j == i:
                    test_X, test_y = X, y
                else:
                    if train_X is None:
                        train_X, train_y = X, y
                    else:
                        train_X, train_y = np.append(train_X, X, axis=0), np.append(train_y, y, axis=0)
            weak_learners, weights = adaboost(train_X, train_y)
            error = compute_error(test_X, test_y, weak_learners, weights)
            errors.append(error)
            print('Decision Stumps: %s \nWeights: %s \nError: %f' % (weak_learners, weights, error))
        print('Errors: %s \nMean Error: %s' % (errors, np.mean(errors)))
    else:
        print('Incorrect mode of operation. Use "erm" or "cv".')
