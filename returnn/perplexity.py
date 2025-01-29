__all__ = ["ExtractPerplexityFromLearningRatesFileJob"]

import ast
from typing import List

from sisyphus import Job, Task, tk



class ExtractPerplexityFromLearningRatesFileJob(Job):
    """
    Extracts the perplexity from the RETURNN learning rates files.
    """

    def __init__(
        self,
        returnn_learning_rates: tk.Path,
        eval_datasets: List[str],
    ):
        self.returnn_learning_rates = returnn_learning_rates
        self.eval_datasets = sorted(eval_datasets)

        self.out_ppl_file = self.output_path("ppl.txt")

        self.out_perplexities = {f"ppl_{d}": self.output_var(f"ppl_{d}") for d in eval_datasets}

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
            full_name = f"{data_set}_loss_ppl" # TODO actually check which name fits
            res.append(f"{data_set} - ppl: {last_entry[full_name]} \n")

        with open(self.out_ppl_file.get_path(), "wt", encoding="utf-8") as f_out:
            f_out.writelines(res)
