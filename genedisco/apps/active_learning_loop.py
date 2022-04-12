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
import sys
import pickle
import inspect
import numpy as np
import slingpy as sp
import importlib.util
from slingpy.utils.path_tools import PathTools
from typing import AnyStr, Dict, List, Optional
from slingpy import AbstractMetric, AbstractBaseModel, AbstractDataSource
from genedisco.evaluation.evaluator import save_top_movers
from genedisco.evaluation.evaluator import Evaluator as Evaluator_HitRatio
from genedisco.active_learning_methods.acquisition_functions.kmeans import Kmeans
from genedisco.active_learning_methods.acquisition_functions.core_set import CoreSet
from genedisco.apps.single_cycle_application import SingleCycleApplication, CustomLoss
from genedisco.active_learning_methods.acquisition_functions.badge_sampling import BadgeSampling
from genedisco.active_learning_methods.acquisition_functions.adversarial_bim import AdversarialBIM
from genedisco.active_learning_methods.acquisition_functions.uncertainty_acquisition import TopUncertainAcquisition
from genedisco.active_learning_methods.acquisition_functions.uncertainty_acquisition import SoftUncertainAcquisition
from genedisco.active_learning_methods.acquisition_functions.margin_sampling_acquisition import \
    MarginSamplingAcquisition
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction
from genedisco.active_learning_methods.acquisition_functions.random_acquisition_function import \
    RandomBatchAcquisitionFunction


class ActiveLearningLoop(sp.AbstractBaseApplication):
    ACQUISITION_FUNCTIONS = [
        "random", "topuncertain", "softuncertain", "marginsample", "coreset", "badge",
        "kmeans_embedding", "kmeans_data", "adversarialBIM", "custom"
    ]

    def __init__(
        self,
        model_name: AnyStr = "randomforest",
        acquisition_function_name: AnyStr = "random",
        acquisition_function_path: AnyStr = "custom",
        acquisition_batch_size: int = 32,
        num_active_learning_cycles: int = 10,
        feature_set_name: AnyStr = SingleCycleApplication.FEATURE_SET_NAMES[0],
        dataset_name: AnyStr = SingleCycleApplication.DATASET_NAMES[1],
        cache_directory: AnyStr = "",
        output_directory: AnyStr = "",
        test_ratio: float = 0.2,
        single_run: bool = True,
        hyperopt_children: bool = False,
        schedule_on_slurm: bool = False,
        schedule_children_on_slurm: bool = False,
        remote_execution_time_limit_days: int = 1,
        remote_execution_mem_limit_in_mb: int = 2048,
        remote_execution_virtualenv_path: AnyStr = "",
        remote_execution_num_cpus: int = 1,
        seed: int = 0
    ):
        self.acquisition_function_name = acquisition_function_name
        self.acquisition_function_path = acquisition_function_path
        PathTools.mkdir_if_not_exists(output_directory)
        self.acquisition_function = ActiveLearningLoop.get_acquisition_function(
            self.acquisition_function_name,
            self.acquisition_function_path
        )
        self.acquisition_batch_size = acquisition_batch_size
        self.num_active_learning_cycles = num_active_learning_cycles
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.model_name = model_name
        self.hyperopt_children = hyperopt_children
        self.test_ratio = test_ratio
        self.cache_directory = cache_directory
        self.schedule_children_on_slurm = schedule_children_on_slurm
        super(ActiveLearningLoop, self).__init__(
            output_directory=output_directory,
            seed=seed,
            evaluate=False,
            hyperopt=False,
            single_run=single_run,
            save_predictions=False,
            schedule_on_slurm=schedule_on_slurm,
            remote_execution_num_cpus=remote_execution_num_cpus,
            remote_execution_time_limit_days=remote_execution_time_limit_days,
            remote_execution_mem_limit_in_mb=remote_execution_mem_limit_in_mb,
            remote_execution_virtualenv_path=remote_execution_virtualenv_path
        )
        
        self.top_movers_filepath = self.prepare_hitratio_evaluation(top_ratio_threshold=0.05)
        
    def prepare_hitratio_evaluation(self, top_ratio_threshold=0.05):
        """Save the top mover genes before AL loop starts to compute the HitRatio metric in next cycles.
        """
        dir_to_save = os.path.join(self.output_directory, "hitratio_artefacts")
        full_path_to_save = save_top_movers(
            top_ratio_threshold=top_ratio_threshold,
            feature_set_name=self.feature_set_name,
            dataset_name=self.dataset_name,
            cache_directory=self.cache_directory,
            test_ratio=self.test_ratio,
            seed=self.seed,
            dir_to_save=dir_to_save
            )
        return full_path_to_save
        
    
    
    @staticmethod
    def get_acquisition_function(
            acquisition_function_name: AnyStr,
            acquisition_function_path: AnyStr
    ) -> BaseBatchAcquisitionFunction:
        if acquisition_function_name == "random":
            return RandomBatchAcquisitionFunction()
        elif acquisition_function_name == "topuncertain":
            return TopUncertainAcquisition()
        elif acquisition_function_name == "softuncertain":
            return SoftUncertainAcquisition()
        elif acquisition_function_name == "marginsample":
            return MarginSamplingAcquisition()
        elif acquisition_function_name == "badge":
            return BadgeSampling()
        elif acquisition_function_name == "coreset":
            return CoreSet()
        elif acquisition_function_name == "kmeans_embedding":
            return Kmeans(representation="linear", n_init=10)
        elif acquisition_function_name == "kmeans_data":
            return Kmeans(representation="raw", n_init=10)
        elif acquisition_function_name == "adversarialBIM":
            return AdversarialBIM()
        elif acquisition_function_name == "custom":
            acqfunc_class = ActiveLearningLoop.get_if_valid_acquisition_function_file(acquisition_function_path)
            return acqfunc_class()
        else:
            raise NotImplementedError()

    @staticmethod
    def get_if_valid_acquisition_function_file(acquisition_function_path: AnyStr):
        if not os.path.exists(acquisition_function_path):
            raise ValueError("The path to the acquisition function file does not exist.")
        else:
            module_name = "custom_acqfunc"
            spec = importlib.util.spec_from_file_location(module_name, acquisition_function_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            custom_acqfunc_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                obj_bases = obj.__bases__
                if BaseBatchAcquisitionFunction in obj_bases:
                    custom_acqfunc_class = obj
            if custom_acqfunc_class is None:
                raise ValueError(f"No valid acquisition function was found at {acquisition_function_path}. "
                                 f"Did you forget to inherit from 'BaseBatchAcquisitionFunction'?")
            return custom_acqfunc_class

    def initialize_pool(self):
        dataset_x = SingleCycleApplication.get_dataset_x(
            self.feature_set_name,
            self.cache_directory
        )
        dataset_y = SingleCycleApplication.get_dataset_y(
            self.dataset_name, 
            self.cache_directory
        )
        available_indices = sorted(
            list(set(dataset_x.get_row_names()).intersection(set(dataset_y.get_row_names())))
        )
        test_indices = sorted(
            list(
                np.random.choice(
                    available_indices, 
                    size=int(self.test_ratio * len(available_indices)), 
                    replace=False)
            )
        )
        available_indices = list(set(available_indices) - set(test_indices))
        return dataset_x, available_indices, test_indices

    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {}

    def get_metrics(self, set_name: AnyStr) -> List[AbstractMetric]:
        return []

    def get_model(self) -> sp.AbstractBaseModel:
        return None

    def train_model(self, model: sp.AbstractBaseModel) -> Optional[sp.AbstractBaseModel]:
        single_cycle_application_args = {
            "model_name": self.model_name,
            "seed": self.seed,
            "remote_execution_time_limit_days": self.remote_execution_time_limit_days,
            "remote_execution_mem_limit_in_mb": self.remote_execution_mem_limit_in_mb,
            "remote_execution_virtualenv_path": self.remote_execution_virtualenv_path,
            "remote_execution_num_cpus": self.remote_execution_num_cpus,
            "schedule_on_slurm": self.schedule_children_on_slurm,
        }
        cumulative_indices = []
        dataset_x, available_indices, test_indices = self.initialize_pool()

        last_selected_indices = sorted(
            list(
                np.random.choice(available_indices, 
                                 size=int(self.acquisition_batch_size),
                                 replace=False)
            )
        )
        cumulative_indices += last_selected_indices
        result_records = list()
        for cycle_index in range(self.num_active_learning_cycles):
            current_cycle_directory = os.path.join(self.output_directory, f"cycle_{cycle_index}")
            PathTools.mkdir_if_not_exists(current_cycle_directory)

            cumulative_indices_file_path = os.path.join(current_cycle_directory, "selected_indices.pickle")
            with open(cumulative_indices_file_path, "wb") as fp:
                pickle.dump(cumulative_indices, fp)
            test_indices_file_path = os.path.join(current_cycle_directory, "test_indices.pickle")
            with open(test_indices_file_path, "wb") as fp:
                pickle.dump(test_indices, fp)

            app = SingleCycleApplication(
                dataset_name=self.dataset_name,
                feature_set_name=self.feature_set_name,
                cache_directory=self.cache_directory,
                output_directory=current_cycle_directory,
                train_ratio=0.8,
                hyperopt=self.hyperopt_children,
                selected_indices_file_path=cumulative_indices_file_path,
                test_indices_file_path=test_indices_file_path,
                top_movers_filepath=self.top_movers_filepath,
                super_dir_to_cycle_dirs=self.output_directory,
                **single_cycle_application_args
            )
            results = app.run().run_result
            result_records.append(results.test_scores)
            available_indices = list(
                set(available_indices) - set(last_selected_indices)
            )

            trained_model_path = results.model_path
            trained_model = app.get_model().load(trained_model_path)

            last_selected_indices = self.acquisition_function(
                dataset_x,
                self.acquisition_batch_size,
                available_indices,
                last_selected_indices,
                trained_model
            )
            cumulative_indices.extend(last_selected_indices)
            cumulative_indices = list(set(cumulative_indices))
            assert len(last_selected_indices) == self.acquisition_batch_size

        results_path = os.path.join(self.output_directory, "results.pickle")
        with open(results_path, "wb") as fp:
            pickle.dump(result_records, fp)
        return None


def main():
    active_learning_loop = sp.instantiate_from_command_line(ActiveLearningLoop)
    active_learning_loop.run()


if __name__ == "__main__":
    main()
