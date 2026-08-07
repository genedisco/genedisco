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
        root_dir = os.path.join(self.read_directory,
                                f"data_{self.dataset_name}",
                                f"feat_{self.feature_set_name}")
        acq_dirs = sorted(d for d in glob.glob(root_dir + "/*") if os.path.isdir(d))
        if not acq_dirs:
            raise FileNotFoundError(
                f"No acquisition-function directories found under {root_dir!r}. "
                f"Check --read_directory, --dataset_name and --feature_set_name "
                f"(they must match the tree written by run_experiments)."
            )

        fig, ax = plt.subplots()
        # Size the palette to the method count, else clrs[i] IndexErrors when
        # there are more methods than colors.
        clrs = sns.color_palette("husl", len(acq_dirs))
        with sns.axes_style("darkgrid"):
            methods = defaultdict(dict)
            plotted_any = False
            max_cycles = 0

            for i, acq_file_name in enumerate(acq_dirs):
                acq_name = os.path.basename(acq_file_name)

                for seed_file_name in sorted(glob.glob(acq_file_name + "/*")):
                    if not os.path.isdir(seed_file_name):
                        continue
                    seed_name = os.path.basename(seed_file_name)

                    scores = []
                    for cycle in range(self.num_cycles):
                        results_file_name = os.path.join(
                            seed_file_name, f"cycle_{cycle}", "test_score.pickle")
                        if not os.path.exists(results_file_name):
                            # Incomplete/short run: stop at the first gap for this seed
                            # instead of killing the whole figure.
                            break
                        with open(results_file_name, "rb") as f:
                            cycle_scores = pickle.load(f)
                        scores.append(cycle_scores[self.metric_name])

                    if scores:
                        methods[acq_name][seed_name] = scores

                seed_curves = list(methods[acq_name].values())
                if not seed_curves:
                    print(f"[warn] no usable cycle results for {acq_name!r}, skipping.")
                    continue

                # Align ragged seeds to their common cycle count before stacking.
                n = min(len(s) for s in seed_curves)
                if n == 0:
                    continue
                y = np.stack([s[:n] for s in seed_curves], axis=0)  # (num_seeds, n)
                mean_y = y.mean(axis=0)
                std_y = y.std(axis=0)
                x = np.arange(n) * self.batch_size
                ax.plot(x, mean_y, label=acq_name.split("acq_")[-1], c=clrs[i])
                ax.fill_between(x, mean_y - std_y, mean_y + std_y,
                                alpha=0.1, facecolor=clrs[i])
                plotted_any = True
                max_cycles = max(max_cycles, n)

            if not plotted_any:
                raise RuntimeError(
                    f"Found {len(acq_dirs)} acquisition dir(s) under {root_dir!r} but no "
                    f"readable 'cycle_*/test_score.pickle' files. Check --num_cycles "
                    f"({self.num_cycles}) and that the runs actually completed."
                )

            ax.legend()
            plt.xticks([int(xi) for xi in np.arange(max_cycles) * self.batch_size])
            ax.set_xlabel("Number of acquired samples")
            ax.set_ylabel(self.metric_name)
            ax.set_title(f"{self.dataset_name} \u00b7 {self.feature_set_name}")

        os.makedirs(self.output_directory, exist_ok=True)
        out_path = os.path.join(self.output_directory,
                                f"{self.dataset_name}_"
                                f"{self.feature_set_name}_"
                                f"{self.batch_size}_"
                                f"{self.metric_name}.pdf")
        plt.savefig(out_path)
        print("Saved plot to", out_path)

def main():
    plot_application = sp.instantiate_from_command_line(PlotApplication)
    plot_application.run()

if __name__ == "__main__":
    main()
