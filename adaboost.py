import argparse

import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


T = 10


def weak_learner(temp_X, temp_y, D):
    m, d = temp_X.shape
    F_star, theta_star, j_star = float('inf'), 0, 0
    for j in range(d):
        sorted_indexes = temp_X[:, j].argsort()
        xj = temp_X[sorted_indexes, j]
        xj = np.append(xj, xj[-1] + 1)
        y_sorted = temp_y[sorted_indexes]
        D_sorted = D[sorted_indexes]
        F = np.sum(D_sorted[np.where(y_sorted == 1)])
        if F < F_star:
            F_star, theta_star, j_star = F, xj[0] - 1, j
        for i in range(m):
            F = F - y_sorted[i] * D_sorted[i]
            if F < F_star and xj[i] != xj[i + 1]:
                F_star, theta_star, j_star = F, (xj[i] + xj[i + 1]) / 2, j
    return theta_star, j_star


def adaboost(temp_X, temp_y, adaboost_t=1):
    m, d = temp_X.shape
    D = np.array([1 / m for i in range(m)])
    weak_learners, errors, weights = [], [], []

    for t in range(adaboost_t):
        weak_learners.append(weak_learner(temp_X, temp_y, D))
        theta, j = weak_learners[-1]
        preds = np.where((theta >= temp_X[:, j]), 1, -1)
        error = np.sum(D[np.where(preds != temp_y)])
        errors.append(error)
        w = 0.5 * math.log(1 / error - 1)
        weights.append(w)
        D = D * np.exp(-w * temp_y * preds)
        D = D / sum(D)
    # print(errors)
    return weak_learners, weights


def compute_error(temp_X, temp_y, weak_learners, weights):
    preds = np.zeros(temp_y.shape)
    for i, learner in enumerate(weak_learners):
        theta, j = learner
        preds = preds + weights[i] * np.where((theta >= temp_X[:, j]), 1, -1)
    preds = np.where(preds < 0, -1, 1)

    return len(np.where(preds != temp_y)[0]) / m


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

        mean_validation_errors, mean_empirical_risks = [], []
        for adaboost_t in range(T):
            empirical_risks, validation_errors = [], []
            for i in range(k):
                print('Executing Fold #: %d' % (i + 1))
                train_X, train_y, test_X, test_y = None, None, None, None
                for j, (X_j, y_j) in enumerate(batches):
                    if j == i:
                        test_X, test_y = X_j, y_j
                    else:
                        if train_X is None:
                            train_X, train_y = X_j, y_j
                        else:
                            train_X, train_y = np.append(train_X, X_j, axis=0), np.append(train_y, y_j, axis=0)
                weak_learners, weights = adaboost(train_X, train_y, adaboost_t)
                empirical_risk = compute_error(X, y, weak_learners, weights)
                validation_error = compute_error(test_X, test_y, weak_learners, weights)
                empirical_risks.append(empirical_risk)
                validation_errors.append(validation_error)
                print('Decision Stumps: %s \nWeights: %s \nError: %f' % (weak_learners, weights, validation_error))
            mean_validation_errors.append(np.mean(validation_errors))
            mean_empirical_risks.append(np.mean(empirical_risks))
            print('T: %d \nErrors: %s \nMean Error: %s' % (adaboost_t, validation_errors, np.mean(validation_errors)))

        # fig, ax = plt.subplots()
        # ax.plot(list(range(T)), mean_validation_errors, '-r', label='Validation Errors')
        # ax.plot(list(range(T)), mean_empirical_risks, '-0', label='Empirical risk on whole dataset')
        # plt.xlabel('T: number of rounds in adaboost')
        # plt.ylabel('Error')
        # ax.legend(loc='upper right')
        # plt.tight_layout()
        # plt.show()

    else:
        print('Incorrect mode of operation. Use "erm" or "cv".')
