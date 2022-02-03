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
import numpy as np
import slingpy as sp
from functools import partial
from slingpy.utils.path_tools import PathTools
from typing import AnyStr, List, Optional, Dict
from slingpy.utils.nestable_pool import NestablePool as Pool
from genedisco.apps.active_learning_loop import ActiveLearningLoop
from slingpy import AbstractMetric, AbstractBaseModel, AbstractDataSource
from genedisco.apps.single_cycle_application import SingleCycleApplication


class RunExperimentsApplication(sp.AbstractBaseApplication):
    def __init__(self,
                 output_directory: AnyStr,
                 cache_directory: AnyStr,
                 model_name: AnyStr = "bayesian_mlp",
                 acquisition_function_path: AnyStr = None,
                 max_num_jobs: int = 90,
                 seed: int = 909,
                 hyperopt_children: bool = True,
                 num_random_seeds: int = 5,
                 num_active_learning_cycles: int = 20,
                 acquisition_batch_size: int = 256,
                 schedule_on_slurm: bool = False,
                 schedule_children_on_slurm: bool = False,
                 remote_execution_time_limit_days: int = 5,
                 remote_execution_mem_limit_in_mb: int = 8048,
                 remote_execution_virtualenv_path: AnyStr = "",
                 remote_execution_num_cpus: int = 1):
        self.cache_directory = cache_directory
        self.model_name = model_name
        self.acquisition_function_path = acquisition_function_path
        self.hyperopt_children = hyperopt_children
        self.schedule_children_on_slurm = schedule_children_on_slurm
        self.max_num_jobs = max_num_jobs
        self.num_random_seeds = num_random_seeds
        self.num_active_learning_cycles = num_active_learning_cycles
        self.acquisition_batch_size = acquisition_batch_size
        super(RunExperimentsApplication, self).__init__(
            output_directory=output_directory,
            evaluate=False,
            hyperopt=False,
            single_run=True,
            save_predictions=False,
            seed=seed,
            schedule_on_slurm=schedule_on_slurm,
            remote_execution_time_limit_days=remote_execution_time_limit_days,
            remote_execution_mem_limit_in_mb=remote_execution_mem_limit_in_mb,
            remote_execution_virtualenv_path=remote_execution_virtualenv_path,
            remote_execution_num_cpus=remote_execution_num_cpus
        )

    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {}

    def get_metrics(self, set_name: AnyStr) -> List[AbstractMetric]:
        return []

    @staticmethod
    def parallel_run_wrapper(args, self_reference):
        seed, baseline, acqfunc_path, dataset, feature_set, model_output_directory = args
        app = ActiveLearningLoop(seed=seed,
                                 num_active_learning_cycles=self_reference.num_active_learning_cycles,
                                 acquisition_batch_size=self_reference.acquisition_batch_size,
                                 acquisition_function_name=baseline,
                                 acquisition_function_path = acqfunc_path,
                                 hyperopt_children=self_reference.hyperopt_children,
                                 dataset_name=dataset,
                                 feature_set_name=feature_set,
                                 schedule_on_slurm=self_reference.schedule_children_on_slurm,
                                 schedule_children_on_slurm=self_reference.schedule_children_on_slurm,
                                 remote_execution_time_limit_days=self_reference.remote_execution_time_limit_days,
                                 remote_execution_mem_limit_in_mb=self_reference.remote_execution_mem_limit_in_mb,
                                 remote_execution_num_cpus=self_reference.remote_execution_num_cpus,
                                 remote_execution_virtualenv_path=self_reference.remote_execution_virtualenv_path,
                                 model_name=self_reference.model_name,
                                 output_directory=model_output_directory,
                                 cache_directory=self_reference.cache_directory)
        app.run()
        return None

    def train_model(self) -> Optional[AbstractBaseModel]:
        acqfunc_path = self.acquisition_function_path
        random_state = np.random.RandomState(self.seed)
        baselines = ActiveLearningLoop.ACQUISITION_FUNCTIONS
        random_seeds = random_state.randint(2**31, size=self.num_random_seeds)
        datasets = SingleCycleApplication.DATASET_NAMES[1:]  # Excluding Shifrut et al.
        feature_sets = SingleCycleApplication.FEATURE_SET_NAMES

        arg_list = []
        for dataset in datasets:
            dataset_output_directory = os.path.join(self.output_directory, f"data_{dataset}")
            PathTools.mkdir_if_not_exists(dataset_output_directory)

            for feature_set in feature_sets:
                feature_output_directory = os.path.join(dataset_output_directory, f"feat_{feature_set}")
                PathTools.mkdir_if_not_exists(feature_output_directory)

                for baseline in baselines:
                    baseline_output_directory = os.path.join(feature_output_directory, f"acq_{baseline}")
                    PathTools.mkdir_if_not_exists(baseline_output_directory)

                    for seed in random_seeds:
                        model_output_directory = os.path.join(baseline_output_directory, f"seed_{seed}")
                        arg_list.append((seed, baseline, acqfunc_path, dataset, feature_set, model_output_directory))

        max_num_processes = self.max_num_jobs
        num_processes = min(len(arg_list), max_num_processes)
        if num_processes == 1:
            outputs = []
            for arg in arg_list:
                outputs.append(RunExperimentsApplication.parallel_run_wrapper(arg, self_reference=self))
        else:
            with Pool(processes=num_processes) as pool:
                outputs = list(pool.imap_unordered(
                    partial(RunExperimentsApplication.parallel_run_wrapper, self_reference=self),
                    arg_list, chunksize=1))
        return None


def main():
    app = sp.instantiate_from_command_line(RunExperimentsApplication)
    results = app.run()


if __name__ == "__main__":
    main()
