import numpy as np
import pandas as pd

from models import TRMF


def ND(prediction, Y, mask=None):
    if mask is None:
        mask = np.array((~np.isnan(Y)).astype(int))
    Y[mask == 0] = 0.
    return abs((prediction - Y) * mask).sum() / abs(Y).sum()


def NRMSE(prediction, Y, mask=None):
    if mask is None:
        mask = np.array((~np.isnan(Y)).astype(int))
    Y[mask == 0] = 0.
    return pow((pow(prediction - Y, 2) * mask).sum(), 0.5) / abs(Y).sum() * pow(mask.sum(), 0.5)


def interpolate_data(data):
    data_interpolated = data.copy()

    N, T = data_interpolated.shape
    total = 0
    for i in range(N):
        start = 0
        end = 0
        move = 'start'
        fill = False

        for t in range(T):
            if (move == 'start') and (np.isnan(data_interpolated[i][start])):
                move = 'end'
                end = start
                if start > 0:
                    a = data_interpolated[i][start-1]
                else:
                    a = 0
            if (move == 'end') and (~np.isnan(data_interpolated[i][end]) or (end == T-1)):
                b = data_interpolated[i][end]
                fill = True

            if fill:
                if np.isnan(b):
                    b = 0
                for j in range(start, end):
                    total += 1
                    data_interpolated[i][j] = a + (j-start)*(b-a)/(end-start)
                fill = False
                move = 'start'
                start = end

            if move == 'start':
                start += 1
            else:
                end += 1

    data_interpolated[np.isnan(data_interpolated)] = 0.
    return data_interpolated


def get_slice(data, T_train, T_test, T_start, normalize=True):
    N = len(data)
    # split on train and test
    train = data[:, T_start:T_start+T_train].copy()
    test = data[:, T_start+T_train:T_start+T_train+T_test].copy()

    # normalize data
    if normalize:
        mean_train = np.array([])
        std_train = np.array([])
        for i in range(len(train)):
            if (~np.isnan(train[i])).sum() == 0:
                mean_train = np.append(mean_train, 0)
                std_train = np.append(std_train, 0)
            else:
                mean_train = np.append(mean_train, train[i][~np.isnan(train[i])].mean())
                std_train = np.append(std_train, train[i][~np.isnan(train[i])].std())
        
        std_train[std_train == 0] = 1.

        train -= mean_train.repeat(T_train).reshape(N, T_train)
        train /= std_train.repeat(T_train).reshape(N, T_train)
        test -= mean_train.repeat(T_test).reshape(N, T_test)
        test /= std_train.repeat(T_test).reshape(N, T_test)
    
    return train, test


def RollingCV(model, data, T_train, T_test, T_step, metric='ND', normalize=True):
    scores = np.array([])
    for T_start in range(0, data.shape[1]-T_train-T_test+1, T_step):
        if isinstance(model, TRMF):
            train, test = get_slice(data, T_train, T_test, T_start, normalize=False)

            input_file = 'rolling_cv_input.csv'
            output_file = 'rolling_cv_output.csv'
            pd.DataFrame(train).to_csv(input_file, na_rep='n', header=False, index=False)

            model.fit(input_file, horizon=T_test, output_file=output_file)
            test_preds = pd.read_csv(output_file, sep=',', header=None).values
        else:
            train, test = get_slice(data, T_train, T_test, T_start, normalize=normalize)
            model.fit(train)
            test_preds = model.predict(T_test)

        if metric == 'ND':
            scores = np.append(scores, ND(test_preds, test))
        if metric == 'NRMSE':
            scores = np.append(scores, NRMSE(test_preds, test))
    return scores
