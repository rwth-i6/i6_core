from sisyphus import Job, Task, tk
from typing import Any, List, Tuple

import numpy as np

from i6_core.util import instanciate_delayed


class PickOptimalParametersJob(Job):
    """
    Pick a set of optimal pickleable parameters based on their assigned (dynamic) score value.
    """

    def __init__(self, parameters: List[Tuple[Any]], values: List[tk.Variable], mode="minimize"):
        """
        :param parameters: list of tuples of parameters, must be pickleable
        :param values: list of tk.Variables containing int or float, used to determine the best
            set of parameters. Some calculations might be done using DelayedOps math.
        :param mode: "minimize" or "maximize"
        """
        assert len(parameters) == len(values)
        for param in parameters[1:]:
            assert len(param) == len(parameters[0]), "all entries should have the same number of parameters"
        assert mode in ["minimize", "maximize"]
        self.parameters = parameters
        self.values = values
        self.mode = mode
        self.num_values = len(values)
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
