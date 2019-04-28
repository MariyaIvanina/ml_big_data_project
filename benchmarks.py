#!/usr/bin/env python3
import argparse

from aggregate import aggregate

from models import TRMF
from models import IndependentFeaturesAutoRegressionModel
from models import AutoRegressionModel
from models import SvdAutoRegressionModel
import utilities
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(prog='Calculating benchmarks with AutoRegression'
                                          ' (independent feature and all together) SVD Autoregression and TRMF.'
                                          ' Takes date range for training and days amount for prediction.')
    parser.add_argument('--data_dir', help='Path to directory with cryptocurrency data')
    parser.add_argument('--begin_date', help="Begin date for training in format 'dd.mm.yyyy'")
    parser.add_argument('--end_date', help="End date for training in format 'dd.mm.yyyy'")
    parser.add_argument('--columns', default='O,C,H,L', help="List of columns to fetch, separated by ','")

    # TRMF params
    # parser.add_argument('--rank', default=32, help="Factorization rank")
    # parser.add_argument('--lambda_x', default=1000, help="Regularization coefficient for inconsistent with"
    #                                                      " autoregression model rows in X matrix")
    # parser.add_argument('--lambda_w', default=1000, help="Regularization coefficient for large autoregression"
    #                                                      " coefficients in W matrix")
    # parser.add_argument('--lambda_f', default=0.01, help="Regularization coefficient for large values in F matrix")
    # parser.add_argument('--eta', default=0.001, help="Regularization coefficient eta for X matrix")

    return parser.parse_args()


def apply_benchmark_model(data_interpolated, model_name):
    T = data_interpolated.shape[1]
    errors_ind_ar = []
    for lags in [[1], [1, 7], [1, 7, 14], [1, 7, 14, 30], [1, 2, 3, 4, 5, 6, 7, 14, 30]]:
        temp = []
        for h in [1, 5, 10, 20]:
            model = model_name(lags=lags)
            scores_nd = utilities.RollingCV(model, data_interpolated, T - h, h, T_step=1, metric='ND')
            scores_nrmse = utilities.RollingCV(model, data_interpolated, T - h, h, T_step=1, metric='NRMSE')
            print('{} performance ND/NRMSE (h = {}, lags = {}): {}/{}'
                  .format(model_name, h, lags,
                          round(np.array(scores_nd).mean(), 3), round(np.array(scores_nrmse).mean(), 3)))
            temp.append((round(np.array(scores_nd).mean(), 3), round(np.array(scores_nrmse).mean(), 3)))
        errors_ind_ar.append(temp)


def _main(args):
    currency_names, currency_values_df = aggregate(args.data_dir, args.begin_date, args.end_date,
                                                   args.columns.split(','))
    data = currency_values_df.T.values
    data_interpolated = utilities.interpolate_data(data)
    apply_benchmark_model(data, TRMF)
    apply_benchmark_model(data_interpolated, IndependentFeaturesAutoRegressionModel)
    apply_benchmark_model(data_interpolated, AutoRegressionModel)
    apply_benchmark_model(data_interpolated, SvdAutoRegressionModel)


if __name__ == '__main__':
    _main(_parse_args())
