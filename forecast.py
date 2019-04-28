#!/usr/bin/env python3
import argparse

from aggregate import aggregate, save_aggregated
from models import TRMF


def _parse_args():
    parser = argparse.ArgumentParser(prog='Applies factorization and forecasts rate for large datasets'
                                          ' of multiple cryptocurrencies rates. Uses TRMF algorithm implementation.'
                                          ' Takes date range for training and days amount for prediction.')
    parser.add_argument('--data_dir', help='Path to directory with cryptocurrency data')
    parser.add_argument('--begin_date', help="Begin date for training in format 'dd.mm.yyyy'")
    parser.add_argument('--end_date', help="End date for training in format 'dd.mm.yyyy'")
    parser.add_argument('--output_file', default='predicted_values.csv', help="File path for predictions")
    parser.add_argument('--output_f_file', default='F.csv', help="File path for F matrix")
    parser.add_argument('--horizon', default=0, help="Days amount to predict")
    parser.add_argument('--columns', default='O,C,H,L', help="List of columns to fetch, separated by ','")
    parser.add_argument('--rank', default=32, help="Factorization rank")
    parser.add_argument('--lags', default='1,2,3,4,5,6,7,14,21', help="List of lags to use in algorithm, separated by comma")
    parser.add_argument('--lambda_x', default=1000, help="Regularization coefficient for inconsistent with"
                                                         " autoregression model rows in X matrix")
    parser.add_argument('--lambda_w', default=1000, help="Regularization coefficient for large autoregression"
                                                         " coefficients in W matrix")
    parser.add_argument('--lambda_f', default=0.01, help="Regularization coefficient for large values in F matrix")
    parser.add_argument('--eta', default=0.001, help="Regularization coefficient eta for X matrix")

    return parser.parse_args()


def _main(args):
    print("Data aggregation ...")
    currency_names, currency_values_df = aggregate(args.data_dir, args.begin_date, args.end_date,
                                                   args.columns.split(','))
    input_data_name = 'tmp_agg'
    save_aggregated(currency_names, currency_values_df, input_data_name)

    print("Compiling library sources ...")
    TRMF.compile_sources()

    print("Running TRMF ...")
    model = TRMF(args.rank, args.lags.split(','), args.lambda_x, args.lambda_w, args.lambda_f, args.eta)
    model.fit(input_data_name + '_values.csv', horizon=args.horizon, output_file=args.output_file,
              output_f_file=args.output_f_file)
    print("Finished!")


if __name__ == '__main__':
    _main(_parse_args())
