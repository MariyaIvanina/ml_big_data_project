import subprocess


class TRMF:
    def __init__(self,
                 rank=32,
                 lags=(1, 2, 3, 4, 5, 6, 7, 14, 21),
                 lambda_x=1000,
                 lambda_w=1000,
                 lambda_f=0.01,
                 eta=0.001):
        """
        Applies factorization and forecasts rate for large datasets of multiple cryptocurrencies rates.
        Uses TRMF algorithm implementation.
        :param rank: Factorization rank
        :param lags: List of lags
        :param lambda_x: Regularization coefficient for inconsistent with autoregression model rows in X matrix
        :param lambda_w: Regularization coefficient for large autoregression coefficients in W matrix
        :param lambda_f: Regularization coefficient for large values in F matrix
        :param eta: Regularization coefficient eta for X matrix
        """
        self.rank = rank
        self.lags = lags
        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.lambda_f = lambda_f
        self.eta = eta

    def fit(self,
            input_file,
            forecast=True,
            horizon=20,
            output_file='predicted_values.csv'):
        """

        :param input_file: File with input values for TRMF algorithm. In fact it is Y matrix.
        :param forecast: Flag determining whether to make a prediction.
        :param horizon: Days amount to predict. Set to 0 if 'forecast' is False.
        :param output_file: File path for predictions.
        :return:
        """
        if not forecast:
            horizon = 0

        call_str = f"./trmf --input_file {input_file} --output_file {output_file} --horizon {horizon}" \
            f" --k {self.rank} --lags {','.join(self.lags)} --lambda_x {self.lambda_x}" \
            f" --lambda_w {self.lambda_w} --lambda_f {self.lambda_f} --eta {self.eta}"

        subprocess.call(call_str.split())

    @staticmethod
    def compile_sources():
        subprocess.call('make -f Makefile'.split())
