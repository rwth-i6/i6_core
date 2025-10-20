__all__ = ["GetOptimalParametersAsVariableJob"]

from typing import Any, Literal, Sequence, Union

import numpy as np
from sisyphus import Job, Task, tk

from i6_core.util import instanciate_delayed


class GetOptimalParametersAsVariableJob(Job):
    """
    Pick a set of optimal parameters based on their assigned (dynamic) score value.
    Each optimal parameter is outputted individually to be accessible in the Sisyphus manager.

    Can be used to e.g. pick best lm-scale and prior scale to a corresponding ScliteJob.out_wer.
    """

    def __init__(
        self,
        *,
        parameters: Sequence[Sequence[Any]],
        values: Sequence[tk.Variable],
        mode: Union[Literal["maximize"], Literal["minimize"]],
    ):
        """
        :param parameters: parameters[best_idx] will be written to self.out_optimal_parameters
             as Sisyphus output variables.
             parameters[best_idx] (and thus self.out_optimal_parameters) is a sequence of fixed length,
             to allow to index into it.
             Thus, len(parameters[i]) must be the same for all i.
        :param values: best_idx = argmax(values) or argmin(values).
              Must have len(values) == len(parameters).
              Some calculations might be done using DelayedOps math beforehand.
        :param mode: "minimize" or "maximize"
        """
        assert len(parameters) == len(values)
        for param in parameters[1:]:
            assert len(param) == len(parameters[0]), "all entries should have the same number of parameters"
        assert mode in ["minimize", "maximize"]
        self.parameters = parameters
        self.values = values
        self.mode = mode
        self.num_parameters = len(parameters[0])

        self.out_optimal_parameters = [self.output_var("param_%i" % i, pickle=True) for i in range(self.num_parameters)]

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        values = instanciate_delayed(self.values)

        if self.mode == "minimize":
            index = np.argmin(values)
        else:
            index = np.argmax(values)

        best_parameters = self.parameters[index]

        for i, param in enumerate(best_parameters):
            self.out_optimal_parameters[i].set(param)
