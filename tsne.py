from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


class VisualizationTSNE:
    def __init__(self, table_filename="F.csv", headers_filename="tmp_agg_names.csv"):
        self.table_filename = table_filename
        self.headers_filename = headers_filename
        self.F = pd.read_csv(table_filename, sep=',', decimal='.', header=None)
        self.headers = pd.read_csv(headers_filename, header=None)
        self.N, self.rank = self.F.shape

    def __fit_transform(self, group_size=4, verbose=1, perplexity=5, n_iter=4000, init_embedding="pca",
                        metric="cosine"):
        self.group_size = group_size
        self.n = self.N // self.group_size
        fitdata = self.F.values.copy().reshape((self.n, self.group_size * self.rank))
        tsne = TSNE(n_components=2, verbose=verbose, perplexity=perplexity, n_iter=n_iter,
                    init=init_embedding, metric=metric)
        self.transformed = tsne.fit_transform(fitdata)

    def __prepare_plot(self):
        plt.figure(figsize=(16, 9))
        plt.scatter(self.transformed[:, 0], self.transformed[:, 1])
        headers = self.headers.values.reshape((self.n, self.group_size))[:, 0]
        for x, y, l in zip(self.transformed[:, 0], self.transformed[:, 1], headers):
            label = l.split("\\")[3].split(":")[0]
            plt.annotate(label,
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

    def show_transformed(self, group_size=4, verbose=1, perplexity=5, n_iter=4000, init_embedding="pca",
                         metric="cosine"):
        self.__fit_transform(group_size=group_size, verbose=verbose, perplexity=perplexity, n_iter=n_iter,
                             init_embedding=init_embedding, metric=metric)
        self.__prepare_plot()
        plt.show()
