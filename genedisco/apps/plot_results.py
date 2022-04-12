"""
Copyright 2021 Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import glob
import pickle
import numpy as np
import slingpy as sp
import seaborn as sns
from typing import AnyStr
import matplotlib.pyplot as plt
from collections import defaultdict


class PlotApplication:
    def __init__(self, output_directory: AnyStr,
                 read_directory: AnyStr,
                 num_cycles: int = 20,
                 batch_size: int = 128,
                 metric_name: AnyStr = "RootMeanSquaredError",
                 dataset_name: AnyStr = "sanchez_2021_tau",
                 feature_set_name: AnyStr = "achilles"):
        self.output_directory = output_directory
        self.read_directory = read_directory
        self.num_cycles = num_cycles
        self.batch_size = batch_size
        self.metric_name = metric_name
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name

    def run(self):
        fig, ax = plt.subplots()
        clrs = sns.color_palette("husl", 5)
        with sns.axes_style("darkgrid"):
            methods = defaultdict(dict)
            x = np.arange(self.num_cycles)*self.batch_size
            root_dir = os.path.join(self.read_directory, f"data_{self.dataset_name}", f"feat_{self.feature_set_name}")
            for i, acq_file_name in enumerate(glob.glob(root_dir + "/*")):
                acq_name = os.path.basename(acq_file_name)

                for seed_file_name in sorted(glob.glob(acq_file_name + "/*")):
                    seed_name = os.path.basename(seed_file_name)

                    for cycle in range(self.num_cycles):
                        results_file_name = os.path.join(seed_file_name, "cycle_" + str(cycle), "test_score.pickle")
                        with open(results_file_name, "rb") as f:
                            test_indices = pickle.load(f)
                        if seed_name not in methods[acq_name]:
                            methods[acq_name][seed_name] = []
                        methods[acq_name][seed_name].append(test_indices[self.metric_name])

                y = np.stack(list(methods[acq_name].values()), axis=-1).T
                mean_y = np.mean(y, axis=0)
                std_y = np.std(y, axis=0)
                ax.plot(x, mean_y, label=acq_name.split("acq_")[1], c=clrs[i])
                ax.fill_between(x, mean_y-std_y, mean_y+std_y, alpha=0.1, facecolor=clrs[i])
            ax.legend()

            xint = []
            for xi in x:
                xint.append(int(xi))
            plt.xticks(xint)

        plt.savefig(os.path.join(self.output_directory,
                                 f"{self.dataset_name}_"
                                 f"{self.feature_set_name}_"
                                 f"{self.batch_size}_"
                                 f"{self.metric_name}.pdf"))


if __name__ == "__main__":
    app = sp.instantiate_from_command_line(PlotApplication)
    app.run()
