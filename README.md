# Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction

This project implements the basic Temporal Regularized Matrix Factorization (TRMF) algorithm. Algorithm was proposed by Hsiang-Fu Yu, Nikhil Rao and Inderjit S. Dhillon in 2016, article can be found here: http://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf.

Implementation was created using C++ programming language. Python was used for benchmarks and data preparation. Eigen 3 library was used for linear algebra methods. 

## About the TRMF algorithm

The TRMF algorithm was developed to solve the problem of time series prediction. The TRFM algorithm should be beneficial at handling multiple time series at once. It also should scale better and allows to restore missing values.

We have `N` time series of length `T` organized into matrix `Y` `(NxT)`. TRMF algorithm finds decomposition `Y ≃ FX`, where `F` is `(Nxk)`, `X` is `(kxT)` and `k` is low. To solve this problem the authors proposed to minimize the following expression:

![equation](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7C%20P_%5COmega%28Y-FX%29%20%5Cright%20%5C%7C%5E%7B2%7D%20&plus;%20%5Clambda%20_f%5Cleft%20%5C%7C%20F%20%5Cright%20%5C%7C%20&plus;%20%5Clambda%20_w%5Cleft%20%5C%7C%20W%20%5Cright%20%5C%7C&plus;%5Clambda%20_x%20%5Csum_%7Br%3D1%7D%5E%7Bk%7DT_%7BAR%7D%28X_k%29%5Crightarrow%20%5Cmathit%7Bmin%7D_%7BF%2CX%2CW%7D)

![equation](https://latex.codecogs.com/gif.latex?T_%7BAR%7D%28X_k%29%3D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bt%3DL&plus;1%7D%5E%7BT%7D%28x_t-%5Csum_%7Bl%5Cin%20LS%7Dw_l%5Ekx_%7Bt-l%7D%29&plus;%5Ceta%20%5Cleft%20%5C%7C%20x%20%5Cright%20%5C%7C%5E2)

Norm operator denotes Frobenius norm.

Optimisation is done step by step separately for `F`, `X` and `W`. For each of these three steps different approach is used.

- `W step`: `W` is updated by solving ridge regression problem using Cholesky factorization.
- `X step`: `X` is updated by using GRAILS algorithm described in [this paper](http://papers.nips.cc/paper/5938-collaborative-filtering-with-graph-information-consistency-and-scalable-methods.pdf).
- `F step`: `F` is updated by using scalable coordinate descent described in [this paper](http://www.cs.utexas.edu/~cjhsieh/icdm-pmf.pdf).

## Installing

Clone the repository. 

## Usage

We suggest using C++ TRMF implementation via python3 script `forecast.py` since it converts and combines CSV files into one file ready to be used in our TRMF implementation. But it also can be used independently.

`forecast.py` usage: `./forecast.py --data_dir data_dir --begin_date beg_date --end_date end_date --columns col --output_file out --output_f_file out_f --rank r --horizon h --lambda_x x --lambda_w w --lambda_f f --eta e --lags lags_list`

To build c++ implementation without using `forecast.py` use: `g++ -std=c++11 -O3 trmf.cpp -o trmf`

`trmf` usage: `./trmf --output_file out --output_f_file out_f --rank r --horizon h --lambda_x x --lambda_w w --lambda_f f --eta e --lags lags_list --input_file in --separator s`

As you can see, both options share a big part of common parameters.

Parameters specific to `forecast.py`:

- `--data_dir data_dir` - Path to directory with cryptocurrency data.
- `--begin_date beg_date` - Begin date for training in format `dd.mm.yyyy`.
- `--end_date end_date` - End date for training in format `dd.mm.yyyy`.
- `--columns col` - List of columns to fetch, separated by `,`.

Parameters for both `forecast.py` and `trmf`:

- `--output_file out` - File path to store predictions.
- `--output_f_file out_f` - File path to store `F` matrix.
- `--rank r` - Factorization rank. `32` by default.
- `--horizon h` - Time ticks amount to predict. `25` by default.
- `--lambda_x x` - Regularization coefficient for inconsistent with autoregression model rows in X matrix. `10000` by default.
- `--lambda_w w` - Regularization coefficient for large autoregression coefficients in W matrix. `1000` by default.
- `--lambda_f f` - Regularization coefficient for large values in F matrix. `0.01` by default.
- `--eta e` - Regularization coefficient eta for X matrix. `0.001` by default.
- `--lags lags_list` - List of lags to use in algorithm, separated by comma without whitespaces. `1,2,3,4,5,6,7,14,21` by default.

Parameters specific to `trmf`:

- `--input_file in` - Location of the input file. (`forecast.py` sets this parameters internally).
- `--separator s` - A character which separates values in the input file. `,` by default.

The prediction is saved in CSV format. Each line corresponds to a single time series.

We prepared a simple example with predefined parameters. It can be executed with `run_example.sh`.

## Data example

An archive with some data can be found [here](https://www.dropbox.com/s/981b9ervs8wve6d/history.tar.gz?dl=0.)
