__all__ = ["ExtractPerplexityFromLearningRatesFileJob"]

import ast
from typing import List

from sisyphus import Job, Task, setup_path, tk



Path = setup_path(__package__)


class ExtractPerplexityFromLearningRatesFileJob(Job):
    """
    Extracts the perplexity from the RETURNN learning rates files.
    """

    def __init__(
        self,
        returnn_learning_rates: tk.Path,
        eval_datasets: List[str],
        loss_names: List[str],
    ):
        self.returnn_learning_rates = returnn_learning_rates
        self.eval_datasets = sorted(eval_datasets)
        self.loss_names = sorted(loss_names)

        self.out_perplexities = self.output_path("ppl.txt")

        self.rqmt = {"gpu": 0, "cpu": 1, "mem": 1, "time": 1}

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        with open(self.returnn_learning_rates.get_path(), "rt", encoding="utf-8") as f_in:
            data = f_in.read()
            lr_dict = ast.literal_eval(data)
            lr_dict = sorted(lr_dict.items(), reverse=True)
            last_entry = lr_dict[0]

        res = []
        for data_set in self.eval_datasets:
            for loss in self.loss_names:
                full_name = f"{data_set}_loss_{loss}"
                res.append(f"{data_set} - {loss}: {last_entry[full_name]} \n")

        with open(self.out_perplexities.get_path(), "wt", encoding="utf-8") as f_out:
            f_out.writelines(res)
