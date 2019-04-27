#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <set>
#include <algorithm>
#include "Eigen/Eigen"
#include <sstream>
#include <cstdlib>
#include <cstring>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using Arr = Eigen::ArrayXXd;

Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");


Vec CholetskyRidge(const Mat& A, const Vec& y, double alpha);
void FStep(const Mat& Y, const Mat& X, const Arr& Omega, double lambda, double epsilon_F, int max_iter_F, Mat& F);
void ConjugateGradient(const Mat& A, const Vec& b, int row, double epsilon_cg, Mat& X);
void WStep(const Mat& X, std::vector<int> lags, double lambda_w, double lambda_x, Mat& W);
std::set<int> GetDeltaSet(const std::set<int>& lags, int d);
std::set<int> GetAllNotEmptyDelta(const std::set<int>& lags);
void ModifyG(const Mat& W, const std::set<int>& lags, int W_idx, Mat& G);
void ModifyD(const Mat& W, const std::set<int>& lags, int W_idx, Mat& D);
void ToLaplacian(Mat& G);
Mat GetFullW(const Mat& W, const std::set<int> lags_set);
void XStep(const Mat& Y, const Arr& Omega, const Mat& W, const Mat& F, const std::set<int> lags_set,
  double lambda_x, double eta, double epsilon_X, int max_iter_X, Mat& X);
void Factorize(const Mat& Y, const Arr& Omega, const std::set<int>& lags_set, int rank, double lambda_f, double lambda_w, double lambda_x, double eta,
  double epsilon_X, double epsilon_F, int max_iter_X, int max_iter_F, int max_global_iter, Mat& F, Mat& X, Mat& W);
void Forecast(Mat& Y, const Arr& Omega, const std::set<int>& lags_set, int rank, int horizon, int T, double lambda_f, double lambda_w, double lambda_x, double eta,
  double epsilon_X, double epsilon_F, int max_iter_X, int max_iter_F, int max_global_iter, Mat& F, Mat& X, Mat& W);


bool parse_config (int argc, char* argv[], char** input_file_name, char** output_file_name, char* delimeter, int* rank, int* horizon, int* T,
  std::set<int>* lags_set, double* lambda_x, double* lambda_w, double* lambda_f, double* eta) {
  if (argc < 2)
    return false;
  char* lags = new char[256];
  int idx = 1;
  while (idx < argc) {
	 std::cout<<argv[idx]<<std::endl;
    if (!strcmp("--input_file", argv[idx]))
    {
      strcpy(*input_file_name, argv[++idx]);
	  idx++;
    }
    else if (!strcmp("--output_file", argv[idx]))
    {
      strcpy(*output_file_name, argv[++idx]);
	  idx++;
    }
    else if (!strcmp("--separator", argv[idx]))
    {
      *delimeter = argv[++idx][0];
	  idx++;
    }
    else if (!strcmp("--k", argv[idx]))
    {
      *rank = atoi(argv[++idx]);
	  idx++;
    }
    else if (!strcmp("--lags", argv[idx]))
    {
      strcpy(lags, argv[++idx]);
	  idx++;
    }
    else if (!strcmp("--horizon", argv[idx]))
    {
      *horizon = atoi(argv[++idx]);
	  idx++;
    }
    else if (!strcmp("--T", argv[idx]))
    {
      *T = atoi(argv[++idx]);
	  idx++;
    }
    else if (!strcmp("--lambda_x", argv[idx]))
    {
      *lambda_x = atof(argv[++idx]);
	  idx++;
    }
    else if (!strcmp("--lambda_w", argv[idx]))
    {
      *lambda_w = atof(argv[++idx]);
	  idx++;
    }
    else if (!strcmp("--lambda_f", argv[idx]))
    {
      *lambda_f = atof(argv[++idx]);
	  idx++;
    }
    else if (!strcmp("--eta", argv[idx]))
    {
      *eta = atof(argv[++idx]);
	  idx++;
    }
    else
      return false;
  }
  std::string lags_string(lags);
  std::istringstream iss(lags_string);
  std::string lag;
  while (std::getline(iss, lag, ',')) {
    lags_set->insert(std::stoi(lag));
  }
  return true;
}

void ReadCSV(char* filename, char delimeter, Mat& Y, Arr& Omega) {

  Y.conservativeResize(0, 0);
  Omega.conservativeResize(0, 0);
  std::ifstream input_stream;
  input_stream.open(filename);
  if (!input_stream.is_open())
	  std::cout<<"Didn't open!\n";
  std::cout<<filename<<std::endl;
  std::string line;
  int row = 0;
  while (std::getline(input_stream, line)) {
	  std::cout<<"Reading...\n";
    Y.conservativeResize(row + 1, Y.cols());
    Omega.conservativeResize(row + 1, Y.cols());

    std::istringstream iss(line);
    std::string value;
    int col = 0;
    while (std::getline(iss, value, delimeter)) {
      if (Y.cols() < col + 1) {
        Y.conservativeResize(Y.rows(), col + 1);
        Omega.conservativeResize(Y.rows(), col + 1);
      }
      try {
        Y(row, col) = std::stod(value);
        Omega(row, col) = 1;
      }
      catch (std::invalid_argument) {
        Y(row, col) = 0;
        Omega(row, col) = 0;
      }
      col += 1;
    }
    row += 1;
  }
  input_stream.close();
}

void Standardize(Mat& M, Vec* means, Vec* scales) {
  *means = M.rowwise().mean();
  M = M.colwise() - *means;
  *scales = (M.array() * M.array()).rowwise().mean().sqrt();
  M = M.array().colwise() / scales->array();
}

int main(int argc, char* argv[]) 
{
  char* input_file_name = new char [256];
  //char* input_file_name = "converted.csv";
  char* output_file_name = new char [256];
  //char* output_file_name = "test.txt";

  char delimeter = ',';
  int rank = 32, horizon = 25, T = -1;
  double lambda_x = 10000, lambda_w = 1000, lambda_f = 0.01, eta = 0.001;
  std::set<int> lags_set = { 1, 2, 3, 4, 5, 6, 7 , 14, 21};

  if (!parse_config(argc, argv, &input_file_name, &output_file_name, &delimeter, &rank, &horizon, &T,
    &lags_set, &lambda_x, &lambda_w, &lambda_f, &eta))
  {
    std::cout << "Error during parsing config file!\n";
    return 1;
  }

  Mat Y;
  Arr Omega;
  ReadCSV(input_file_name, delimeter, Y, Omega);

  int n = Y.rows();
  if (T == -1) T = Y.cols();
  std::cout<<n<<"\t"<<T<<std::endl;
  double epsilon_X = 0.0001;
  double epsilon_F = 0.0001;
  int max_iter_X = 10;
  int max_iter_F = 10;
  int max_global_iter = 20;

  Mat Y_pred = Y.leftCols(T);
  Arr Omega_part = Omega.leftCols(T);

  Mat F, X;
  Mat W(rank, lags_set.size());
  Forecast(Y_pred, Omega_part, lags_set, rank, horizon, T, lambda_f, lambda_w, lambda_x, eta, epsilon_X, epsilon_F,
    max_iter_X, max_iter_F, max_global_iter, F, X, W);

  std::ofstream output_file(output_file_name);
  output_file << Y_pred.format(CSVFormat);
  output_file.close();

  std::ofstream F_out_file("F.csv");
  F_out_file << F.format(CSVFormat);
  F_out_file.close();
  std::cout<<"Thank God, it's done!\n";
  return 0;
}

void Forecast(Mat& Y, const Arr& Omega, const std::set<int>& lags_set, int rank, int horizon, int T, double lambda_f, double lambda_w, double lambda_x, double eta,
  double epsilon_X, double epsilon_F, int max_iter_X, int max_iter_F, int max_global_iter, Mat& F, Mat& X, Mat& W) {

  int n = Y.rows();

  Vec means, scales;
  Standardize(Y, &means, &scales);

  Y = Y.array() * Omega;

  Vec reference_column = Y.col(Y.cols() - 1);

  Factorize(Y, Omega, lags_set, rank, lambda_f, lambda_w, lambda_x, eta, epsilon_X, epsilon_F, max_iter_X, max_iter_F, max_global_iter, F, X, W);

  X.conservativeResize(rank, T + horizon);

  std::vector<int> lags_vec(lags_set.begin(), lags_set.end());
  std::sort(lags_vec.begin(), lags_vec.end());

  for (int row = 0; row<rank; ++row) {
    for (int t = T; t<T + horizon; ++t) {
      double value = 0;
      for (int lag_idx = 0; lag_idx<lags_vec.size(); ++lag_idx) {
        value += X(row, t - lags_vec[lag_idx]) * W(row, lag_idx);
      }
      X(row, t) = value;
    }
  }
  Mat X_pred = X.rightCols(horizon + 1);
  Mat Y_pred = F * X_pred;
  Y = Y_pred.rightCols(horizon);
}

void Factorize(const Mat& Y, const Arr& Omega, const std::set<int>& lags_set, int rank, double lambda_f, double lambda_w, double lambda_x, double eta,
  double epsilon_X, double epsilon_F, int max_iter_X, int max_iter_F, int max_global_iter, Mat& F, Mat& X, Mat& W) 
{

  std::vector<int> lags_vec(lags_set.begin(), lags_set.end());

  // Using the SVD decomposition as first approximations for F and X 

  Eigen::BDCSVD<Mat> svd(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);

  F = svd.matrixU().leftCols(rank) * svd.singularValues().head(rank).asDiagonal();
  X = svd.matrixV().leftCols(rank).transpose();

  for (int iter = 0; iter<max_global_iter; ++iter) {
    std::cout << "Iteration " << iter + 1 << std::endl;
    Mat X_prev = X;
    WStep(X, lags_vec, lambda_w, lambda_x, W);
    XStep(Y, Omega, W, F, lags_set, lambda_x, eta, epsilon_X, max_iter_X, X);
    FStep(Y, X, Omega, lambda_f, epsilon_F, max_iter_F, F);

    double diff_X_norm = (X - X_prev).norm();
    if ((diff_X_norm) < epsilon_X * X.rows() * X.cols()) {
      return;
    }
  }
}


Vec CholetskyRidge(const Mat& A, const Vec& y, double alpha) 
{
  Eigen::LDLT<Mat> ldlt;
  Mat I(A.cols(), A.cols());
  I.setIdentity();
  ldlt.compute(A.transpose() * A + alpha * I);
  Eigen::VectorXd w_r = ldlt.solve(A.transpose() * y);
  return w_r;
}


// Coordinate descent from http://www.cs.utexas.edu/~cjhsieh/icdm-pmf.pdf
void FStep(const Mat& Y, const Mat& X, const Arr& Omega, double lambda, double epsilon_F, int max_iter_F, Mat& F) 
{
  Mat R = Y - F * X;
  for (int i = 0; i < max_iter_F; ++i) {
    double delta_sq = 0;
    for (int row = 0; row < F.rows(); ++row) {
      for (int col = 0; col < F.cols(); ++col) {
        double denom = lambda + (X.row(col).array() * X.row(col).array() * Omega.row(row).array()).matrix().sum();
        double numer = (X.row(col).array() * (R.row(row).array() + F(row, col) * X.row(col).array()) * Omega.row(row).array()).matrix().sum();
        double f = numer / denom;
        R.row(row).array() -= (f - F(row, col)) * X.row(col).array();
        delta_sq += (f - F(row, col)) * (f - F(row, col));
        F(row, col) = f;
      }
    }
    if (delta_sq < epsilon_F * F.rows() * F.cols()) {
      return;
    }
  }
}

// Conjugate gradient method taken from https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
void ConjugateGradient(const Mat& A, const Vec& b, int row, double epsilon_cg, Mat& X) 
{
  Vec r0 = b - A * X.row(row).transpose();
  Vec r1 = r0;
  Vec p = r0;
  for (int k = 0; k<A.rows(); ++k) {
    double alpha = r0.dot(r0) / (p.transpose() *  A * p);
    X.row(row) += (alpha * p).transpose();
    r1 = r0 - alpha * A * p;
    if (r1.norm() < epsilon_cg * X.cols()) {
      return;
    }
    double beta = r1.dot(r1) / r0.dot(r0);
    p = r1 + beta * p;
    r0 = r1;
  }
  return;
}

void WStep(const Mat& X, std::vector<int> lags, double lambda_w, double lambda_x, Mat& W) 
{
  std::sort(lags.begin(), lags.end());
  int T = X.cols();
  int k = X.rows();
  int l = lags.size();
  int L = lags.back();
  double alpha = 0.5 * lambda_w / lambda_x;

  for (int row = 0; row < k; ++row) {
    Mat M(T - L, l);
    Vec y(T - L);
    for (int i = L; i<T; ++i) {
      y(i - L) = X(row, i);
      for (int j = 0; j<l; ++j) {
        M(i - L, l - 1 - j) = X(row, i - lags[j]);
      }
    }
    W.row(row) = CholetskyRidge(M, y, alpha).transpose();
  }
}


std::set<int> GetDeltaSet(const std::set<int>& lags, int d) 
{
  std::set<int> deltaset;
  int lag = 0;
  if (lag - d == 0) {
    deltaset.insert(lag);
  }
  for (int lag : lags) {
    if (lag - d == 0) {
      deltaset.insert(lag);
    }
    else if (lags.find(lag - d) != lags.end()) {
      deltaset.insert(lag);
    }
  }
  return deltaset;
}


std::set<int> GetAllNotEmptyDelta(const std::set<int>& lags) 
{
  std::set<int> deltas;
  int delta = 0;
  deltas.insert(delta);

  for (int lag1 : lags) {
    int lag2 = 0; {
      delta = lag1 - lag2;
      if (delta >= 0) {
        deltas.insert(delta);
      }
    }
  }

  for (int lag1 : lags) {
    for (int lag2 : lags) {
      delta = lag1 - lag2;
      if (delta >= 0) {
        deltas.insert(delta);
      }
    }
  }
  return deltas;
}


void ModifyG(const Mat& W, const std::set<int>& lags, int W_idx, Mat& G) 
{
  double w_0 = -1;
  int T = G.cols();
  int L = *(std::max_element(lags.begin(), lags.end()));
  int m = 1 + L;
  std::set<int> deltas = GetAllNotEmptyDelta(lags);

  for (int idx = 0; idx<G.rows(); ++idx) {
    int t = idx + 1;
    for (int d : deltas) {
      if ((d + idx) < G.cols()) {
        std::set<int> deltaset = GetDeltaSet(lags, d);
        double value = 0.0;
        for (int l : deltaset) {
          if ((m <= (t + l)) && ((t + l) <= T)) {
            if ((l - d == 0) && (l == 0)) {
              value += -w_0 * w_0;
            }
            else if (l - d == 0) {
              value += -W(W_idx, l - 1) * w_0;
            }
            else if (l == 0) {
              value += -w_0 * W(W_idx, l - d - 1);
            }
            else {
              value += -W(W_idx, l - 1) * W(W_idx, l - d - 1);
            }
          }
        }
        G(t - 1, t + d - 1) = value;
        G(t + d - 1, t - 1) = value;
      }
    }
  }
}

void ModifyD(const Mat& W, const std::set<int>& lags, int W_idx, Mat& D) 
{
  double w_0 = -1;
  int T = D.cols();
  int L = *(std::max_element(lags.begin(), lags.end()));
  int m = 1 + L;
  double w_sum = 0.0;
  std::set<int> lags_hat = lags;
  lags_hat.insert(0);

  w_sum += w_0;
  for (int l : lags) {
    w_sum += W(W_idx, l - 1);
  }

  for (int idx = 0; idx<D.rows(); ++idx) {
    int t = idx + 1;
    double value = 0.0;
    for (int l : lags_hat) {
      if ((m <= (t + l)) && ((t + l) <= T)) {
        if (l == 0) {
          value += w_sum * w_0;
        }
        else {
          value += w_sum * W(W_idx, l - 1);
        }
      }
      D(t - 1, t - 1) = value;
    }
  }
}

void ToLaplacian(Mat& G) 
{
  G *= -1;
  G -= G.colwise().sum().asDiagonal();
}

Mat GetFullW(const Mat& W, const std::set<int> lags_set) 
{
  std::vector<int> lags_vec(lags_set.begin(), lags_set.end());
  std::sort(lags_vec.begin(), lags_vec.end());

  Mat W_full(W.rows(), lags_vec.back());
  W_full.setZero();
  for (int col = 0; col<W.cols(); ++col) {
    W_full.col(lags_vec[col] - 1) = W.col(col);
  }
  return W_full;
}


void XStep(const Mat& Y, const Arr& Omega, const Mat& W, const Mat& F, const std::set<int> lags_set,
  double lambda_x, double eta, double epsilon_X, int max_iter_X, Mat& X) 
{
  Mat W_full = GetFullW(W, lags_set);

  int T = Y.cols();

  Mat I(T, T);
  I.setIdentity();

  for (int iter = 0; iter<max_iter_X; ++iter) {
    Mat X_prev = X;
    for (int row = 0; row<X.rows(); ++row) {
      Mat G(T, T);
      Mat D(T, T);
      G.setZero();
      D.setZero();

      ModifyG(W_full, lags_set, row, G);
      ModifyD(W_full, lags_set, row, D);

      G.diagonal().setZero();
      ToLaplacian(G);

      G *= 0.5;
      G += 0.5 * eta * I + 0.5 * D;
      G *= lambda_x;

      Mat Y_tilde(Y.rows(), Y.cols());
      Y_tilde = Y - F * X + F.col(row) * X.row(row);

      Vec b(T);

      for (int j = 0; j<T; ++j) {
        b(j) = (F.col(row).array() * F.col(row).array() * Omega.col(j)).sum();
      }

      G.diagonal() += b;

      Vec lhs(T);
      lhs = Y.transpose() * F.col(row);
      ConjugateGradient(G, lhs, row, 0.0001, X);
    }
    if ((X - X_prev).norm() < epsilon_X * X.rows() * X.cols()) {
      break;
    }
  }
}
