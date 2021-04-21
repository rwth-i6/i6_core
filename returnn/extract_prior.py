__all__ = ["ExtractPriorFromHDF5Job"]

import math

import numpy as np
import h5py

from sisyphus import *

Path = setup_path(__package__)


class ExtractPriorFromHDF5Job(Job):
    def __init__(self, returnn_model, layer="output", plot_prior=False):
        self.returnn_model = returnn_model
        self.layer = layer
        self.plot_prior = plot_prior

        self.prior = self.output_path("prior.xml", cached=True)
        if self.plot_prior:
            self.prior_plot = self.output_path("prior.png")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        model = h5py.File(tk.uncached_path(self.returnn_model), "r")
        priors_set = model["%s/priors" % self.layer]

        priors_list = np.asarray(priors_set[:])

        with open(self.prior.get_path(), "wt") as out:
            out.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n'
                % priors_list.shape[0]
            )
            out.write(" ".join("%.20e" % math.log(s) for s in priors_list) + "\n")
            out.write("</vector-f32>")

        if self.plot_prior:
            import matplotlib.pyplot as plt

            xdata = range(len(priors_list))
            plt.semilogy(xdata, priors_list)

            plt.xlabel("emission idx")
            plt.ylabel("prior")
            plt.grid(True)
            plt.savefig(self.prior_plot.get_path())
