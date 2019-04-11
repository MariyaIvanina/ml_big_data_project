# Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction

This project is implementing of the basic Temporal Regularized Matrix Factorization (TRMF) algorithm. Algorithm was purposed by by Hsiang-Fu Yu, Nikhil Rao and Inderjit S. Dhillon in 2016, article could be found here: http://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf.

Implementation was created using python programming language.

## About the TRMF algorithm

The TRMF algorithm is created to solve the problem of timeseries prediction along with ARIMA and DLM. The features of the algorithm that distinguish it from others used in similar tasks are scalability and flexibility for handling missing values.

We have N timeseries of length T and they organized into matrix Y (NxT). In general we want to find Y ≃ FX, where F is (Nxk), X is (kxT) and k is low. To solve this problem the authors proposed to minimize the following expression:

![equation](https://latex.codecogs.com/gif.latex?%24%24%5Cmin%5Climits_%7BF%2CX%7D%5Csum%5Climits_%7B%28i%2Ct%29%5Cin%5COmega%7D%5Cleft%28Y_%7Bit%7D-f_i%5ETx_t%5Cright%29%5E2&plus;%5Clambda_fR_f%28F%29&plus;%5Clambda_xR_x%28X%29.%24%24)

## Implementation

For comparison with TRMF various autoregression algorithms were implemented, where ND and NRMSE metrics were used. Results could be found in Crypto_data_investigation.ipynb file. Realisation of TRMF stored in file trmf.py.
