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
    parser.add_argument('--begin_date', help="Begin date in format 'yyyy.mm.dd'")
    parser.add_argument('--end_date', help="End date in format 'yyyy.mm.dd'")
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
    :return: List of names for time series names and dataframe with their values.
            List is named index for columns in dataframe.
    """
    time_series_index = []
    time_series_df = None

    folders = os.listdir(data_dir)

    for i, folder in enumerate(folders):
        filepath = _get_filepath(data_dir, folders[i])
        currency = filepath.split('/')[-1].split('_')[0]

        data = pd.read_csv(filepath, delimiter=';', decimal=',')
        data['T'] = data['T'].map(lambda string: datetime.datetime.strptime(string, '%Y-%m-%dT%H:%M:%S'))

        if data['T'].iloc[0] > begin_date or data['T'].iloc[-1] < end_date:
            print("File {} doesnt contain all data for the period of interest".format(filepath))
            continue

        data = data[(data['T'] >= begin_date) & (data['T'] <= end_date)]
        data.set_index('T', inplace=True)
        data = data[columns]

        series_names = ['{}:{}'.format(currency, colname) for colname in data.columns]
        data.columns = series_names

        time_series_index.extend(series_names)

        time_series_df = data if i == 0 else pd.merge(time_series_df, data,
                                                      how='outer', left_index=True, right_index=True)

    return time_series_index, time_series_df


def _main(args):
    begin_date = datetime.datetime.strptime(args.begin_date, '%d.%m.%Y')
    end_date = datetime.datetime.strptime(args.end_date, '%d.%m.%Y')
    index, time_series = aggregate(args.data_dir, begin_date, end_date, args.columns.split(','))

    index_file_name = args.output_filename + '_index.csv'
    pd.Series(index).to_csv(index_file_name, index=False)

    series_file_name = args.output_filename + '_series.csv'
    time_series.T.to_csv(series_file_name, na_rep='n', header=False, index=False)


if __name__ == '__main__':
    _main(_parse_args())
