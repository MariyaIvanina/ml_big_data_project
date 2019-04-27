#!/usr/bin/env python3
import os
from os.path import join

import pandas as pd
import datetime
import argparse


def _parse_args():
    parser = argparse.ArgumentParser(prog='Aggregates cryptocurrency data from all subdirectories to two files with'
                                          ' indexes and series. Data is filtered according to chosen dates range.')
    parser.add_argument('--data_dir', help='Path to directory with cryptocurrency data')
    parser.add_argument('--begin_date', help="Begin date in format 'dd.mm.yyyy'")
    parser.add_argument('--end_date', help="End date in format 'dd.mm.yyyy'")
    parser.add_argument('--columns', default='O,C,H,L', help="List of columns to fetch, separated by ','")
    parser.add_argument('--output_filename', default='aggregated',
                        help='Common name for output files. Pass without extension')
    return parser.parse_args()


def _get_filepath(data_dir, foldername, time='day'):
    filename = '{}_{}_1.csv'.format(foldername, time)
    return join(data_dir, foldername, time, filename)


def aggregate(data_dir, begin_date, end_date, columns):
    """
    Aggregates cryptocurrency data from all subdirectories in 'data_dir'.
    Data is filtered according to chosen dates range.
    :param data_dir: Path to directory with cryptocurrency data
    :param begin_date: Begin date in format 'yyyy.mm.dd'
    :param end_date: End date in format 'yyyy.mm.dd'
    :param columns: List of columns to fetch
    :return: List of time series names and dataframe with time series values.
            Each name corresponds to appropriate column in dataframe. Dataframe has dates as index.
    """
    time_series_names = []
    time_series_df = None

    folders = os.listdir(data_dir)

    for i, folder in enumerate(folders):
        filepath = _get_filepath(data_dir, folder)
        currency = filepath.split('/')[-1].split('_')[0]

        data = pd.read_csv(filepath, delimiter=';', decimal=',')
        data['T'] = data['T'].map(lambda string: datetime.datetime.strptime(string, '%Y-%m-%dT%H:%M:%S'))

        if data['T'].iloc[0] > begin_date or data['T'].iloc[-1] < end_date:
            print("File {} doesn't contain all data for the period of interest".format(filepath))
            continue

        data = data[(data['T'] >= begin_date) & (data['T'] <= end_date)]
        data.drop_duplicates(subset=['T'], inplace=True)
        data.set_index('T', inplace=True)
        data = data[columns]

        series_names = ['{}:{}'.format(currency, colname) for colname in data.columns]
        data.columns = series_names

        time_series_names.extend(series_names)

        time_series_df = data if i == 0 else pd.merge(time_series_df, data,
                                                      how='outer', left_index=True, right_index=True)

    return time_series_names, time_series_df


def save_aggregated(time_series_names, time_series_df, output_filename):
    """
    Saves aggreagated time series names, values and dates to separate files with.
    :param time_series_names: names for time series
    :param time_series_df: Time series dataframe.
    :param output_filename: Common name for output files.
    """
    ts_names_filename = output_filename + '_names.csv'
    pd.Series(time_series_names).to_csv(ts_names_filename, index=False)

    ts_values_filename = output_filename + '_values.csv'
    time_series_df.T.to_csv(ts_values_filename, na_rep='n', header=False, index=False)

    ts_dates_filename = output_filename + '_dates.csv'
    time_series_df.index.to_series().to_csv(ts_dates_filename)


def _main(args):
    begin_date = datetime.datetime.strptime(args.begin_date, '%d.%m.%Y')
    end_date = datetime.datetime.strptime(args.end_date, '%d.%m.%Y')

    ts_names, ts_values_df = aggregate(args.data_dir, begin_date, end_date, args.columns.split(','))
    save_aggregated(ts_names, ts_values_df, args.output_filename)


if __name__ == '__main__':
    _main(_parse_args())
