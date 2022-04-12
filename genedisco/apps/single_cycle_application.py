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
import torch
import pickle
import slingpy as sp
from slingpy.models import torch_model
from slingpy.evaluation.evaluator import Evaluator
from sklearn.ensemble import RandomForestRegressor
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource
from genedisco.models import meta_models
from genedisco.evaluation import hitratio
from genedisco.models import pytorch_models
from genedisco.datasets.features.achilles import Achilles
from typing import Any, AnyStr, Dict, List, Tuple, Union, Optional
from genedisco.datasets.features.string_embedding import STRINGEmbedding
from genedisco.evaluation.evaluator import Evaluator as Evaluator_HitRatio
from genedisco.datasets.screens.zhuang_2019_nk_cancer import Zhuang2019NKCancer
from genedisco.datasets.screens.schmidt_2021_t_cells_il2 import Schmidt2021TCellsIL2
from genedisco.datasets.screens.sanchez_2021_neurons_tau import Sanchez2021NeuronsTau
from genedisco.datasets.screens.schmidt_2021_t_cells_ifng import Schmidt2021TCellsIFNg
from genedisco.datasets.features.ccle_protein_quantification import CCLEProteinQuantification
from genedisco.datasets.screens.zhu_2021_sarscov2_host_factors import Zhu2021SARSCoV2HostFactors


SklearnRandomForestRegressor = meta_models.SklearnRandomForestRegressor


def update_dictionary_keys_with_prefixes(input_dict: Dict[AnyStr, Any],
                                         prefix: AnyStr):
    """Adds a prefix to the keys of a dictionary."""
    output_dict = dict(
        (prefix + key, value) for (key, value) in input_dict.items()
    )
    return output_dict


class CustomLoss(sp.TorchLoss):
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self,
                 y_pred: List[torch.Tensor],
                 y_true: List[torch.Tensor]
                 ) -> torch.Tensor:
        loss = self.loss_fn(y_pred[0], y_true[0].float())
        return loss


class SingleCycleApplication(sp.AbstractBaseApplication):
    DATASET_NAMES = [
        "shifrut_2018", "schmidt_2021_ifng",
        "schmidt_2021_il2", "zhuang_2019_nk",
        "sanchez_2021_tau", "zhu_2021_sarscov2"
    ]

    FEATURE_SET_NAMES = ["achilles", "ccle", "string"]

    def __init__(
        self,
        dataset_name: AnyStr = DATASET_NAMES[0],
        feature_set_name: AnyStr = FEATURE_SET_NAMES[0],
        output_directory: AnyStr = "",
        cache_directory: AnyStr = "",
        schedule_on_slurm: bool = False,
        remote_execution_time_limit_days: int = 3,
        remote_execution_mem_limit_in_mb: int = 2048,
        remote_execution_virtualenv_path: AnyStr = "",
        remote_execution_num_cpus: int = 1,
        split_index_outer: int = 0,
        split_index_inner: int = 0,
        num_splits_outer: int = 2,
        num_splits_inner: int = 2,
        model_name: AnyStr = "mlp",
        evaluate_against: AnyStr = "test",
        selected_indices_file_path: AnyStr = "",
        test_indices_file_path: AnyStr = "",
        train_ratio: float = 0.8,
        single_run: bool = True,
        hyperopt: bool = False,
        num_hyperopt_runs: int = 15,
        hyperopt_offset: int = 0,
        hyperopt_metric_name: AnyStr = "MeanAbsoluteError",
        train: bool = True,
        rf_max_depth: int = -1,  # randomforest hyperparams
        rf_num_estimators: int = 100,  # ensemble_model_hyperparms
        dn_num_layers: int = 2,  # deep net hyperparams
        dn_hidden_layer_size: int = 16,  # deep net hyperparams
        top_movers_filepath: AnyStr = "",
        super_dir_to_cycle_dirs: AnyStr = "",
        seed: int = 0
    ):
        self.model_hyperparams = {
            "rf_max_depth": rf_max_depth,
            "rf_num_estimators": rf_num_estimators,
            "dn_num_layers": dn_num_layers,
            "dn_hidden_layer_size": dn_hidden_layer_size,
        }
        self.model = None
        self.rf_max_depth = rf_max_depth
        self.rf_num_estimators = rf_num_estimators
        self.dn_num_layers = dn_num_layers
        self.dn_hidden_layer_size = dn_hidden_layer_size
        self.model_name = model_name
        self.train = train
        self.cache_directory = cache_directory
        self.test_indices_file_path = test_indices_file_path
        self.selected_indices_file_path = selected_indices_file_path
        self.train_ratio = train_ratio
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.hyperopt_offset = hyperopt_offset
        self.hyperopt_metric_name = hyperopt_metric_name
        self.seed = seed
        self.top_movers_filepath = top_movers_filepath
        self.super_dir_to_cycle_dirs = super_dir_to_cycle_dirs

        with open(self.test_indices_file_path, "rb") as fp:
            self.test_indices = pickle.load(fp)
        with open(self.selected_indices_file_path, "rb") as fp:
            self.selected_indices = pickle.load(fp)

        super(SingleCycleApplication, self).__init__(
            output_directory=output_directory,
            schedule_on_slurm=schedule_on_slurm,
            split_index_outer=split_index_outer,
            split_index_inner=split_index_inner,
            num_splits_outer=num_splits_outer,
            num_splits_inner=num_splits_inner,
            evaluate_against=evaluate_against,
            nested_cross_validation=False,
            save_predictions=False,
            save_predictions_file_format="tsv",
            hyperopt=hyperopt,
            single_run=single_run,
            num_hyperopt_runs=num_hyperopt_runs,
            seed=seed,
            remote_execution_time_limit_days=remote_execution_time_limit_days,
            remote_execution_virtualenv_path=remote_execution_virtualenv_path,
            remote_execution_num_cpus=remote_execution_num_cpus,
            remote_execution_mem_limit_in_mb=remote_execution_mem_limit_in_mb
        )

    def get_metrics(self, set_name: AnyStr) -> List[sp.AbstractMetric]:
        return [
            sp.metrics.MeanAbsoluteError(),
            sp.metrics.RootMeanSquaredError(),
            sp.metrics.SymmetricMeanAbsolutePercentageError(),
            sp.metrics.SpearmanRho(),
            hitratio.HitRatio()
        ]
    
    def evaluate_model(self, model: AbstractBaseModel, dataset_x: AbstractDataSource, dataset_y: AbstractDataSource,
                       with_print: bool = True, set_name: AnyStr = "", threshold=None) \
            -> Dict[AnyStr, Union[float, List[float]]]:
        """
        Evaluates model performance.
        Because of the HitRatio metric does not follow the same pattern of the supervised learning evaluation
        implemented by Slingpy, The evaluate_method of the AbstractBaseApplication is overridden here so that
        the HitRatio metric uses a different customized Evaluator that is implemented at GeneDisco level.

        Args:
            model: The model to evaluate.
            dataset: The dataset used for evaluation.
            with_print: Whether or not to print results to stdout.
            set_name: The name of the dataset being evaluated.
            threshold: An evaluation threshold (derived in sample) for discrete classification metrics,
             or none if a threshold should automatically be selected.

        Returns:
            A dictionary with each entry corresponding to an evaluation metric with one or more associated values.
        """
        all_metrics = self.get_metrics("Test set")
        all_metric_names = [metric.__class__.__name__ for metric in all_metrics]
        if "HitRatio" in all_metric_names:
            hitratio_metric_dic = Evaluator_HitRatio.evaluate(
                top_movers_filepath=self.top_movers_filepath,
                super_dir_to_cycle_dirs=self.super_dir_to_cycle_dirs,
                metrics=[metric for metric in all_metrics if metric.__class__.__name__ == "HitRatio"],
            )
            all_metrics_except_hitratio = [metric for metric in all_metrics if metric.__class__.__name__ != "HitRatio"]
            other_metrics_dic = Evaluator.evaluate(model, dataset_x, dataset_y, all_metrics_except_hitratio,
                                  with_print=with_print, set_name=set_name, threshold=threshold)
            all_metrics_dic = {**hitratio_metric_dic, **other_metrics_dic}
        else:
            all_metrics_dic = Evaluator.evaluate(model, dataset_x, dataset_y, all_metrics,
                                  with_print=with_print, set_name=set_name, threshold=threshold)

        return all_metrics_dic

    @staticmethod
    def get_dataset_y(dataset_name, cache_directory):
        if dataset_name == SingleCycleApplication.DATASET_NAMES[0]:
            # dataset_y = Shifrut2018TCells.load_data(cache_directory)
            raise NotImplementedError("Shifrut et al is currently not available for automated evaluation.")
        elif dataset_name == SingleCycleApplication.DATASET_NAMES[1]:
            dataset_y = Schmidt2021TCellsIFNg.load_data(cache_directory)
        elif dataset_name == SingleCycleApplication.DATASET_NAMES[2]:
            dataset_y = Schmidt2021TCellsIL2.load_data(cache_directory)
        elif dataset_name == SingleCycleApplication.DATASET_NAMES[3]:
            dataset_y = Zhuang2019NKCancer.load_data(cache_directory)
        elif dataset_name == SingleCycleApplication.DATASET_NAMES[4]:
            dataset_y = Sanchez2021NeuronsTau.load_data(cache_directory)
        elif dataset_name == SingleCycleApplication.DATASET_NAMES[5]:
            dataset_y = Zhu2021SARSCoV2HostFactors.load_data(cache_directory)
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented.")
        return dataset_y

    @staticmethod
    def get_dataset_x(feature_set_name, cache_directory):
        if feature_set_name == SingleCycleApplication.FEATURE_SET_NAMES[0]:
            dataset = Achilles.load_data(cache_directory)
        elif feature_set_name == SingleCycleApplication.FEATURE_SET_NAMES[1]:
            dataset = CCLEProteinQuantification.load_data(cache_directory)
        elif feature_set_name == SingleCycleApplication.FEATURE_SET_NAMES[2]:
            dataset = STRINGEmbedding.load_data(cache_directory)
        else:
            raise NotImplementedError()

        dataset_x = sp.CompositeDataSource([dataset])
        return dataset_x

    def load_data(self) -> Dict[AnyStr, sp.AbstractDataSource]:
        dataset_x = SingleCycleApplication.get_dataset_x(self.feature_set_name, self.cache_directory)
        dataset_y = SingleCycleApplication.get_dataset_y(self.dataset_name, self.cache_directory)

        # Subset dataset_y by the overlap of genes present in both dataset_x and dataset_y.
        avail_names = sorted(
            list(set(dataset_x.get_row_names()).intersection(
                 set(dataset_y.get_row_names())))
        )
        dataset_y = dataset_y.subset(avail_names)
        dataset_x = dataset_x.subset(avail_names)

        dataset_y = dataset_y.subset(self.selected_indices)

        stratifier = sp.StratifiedSplit()
        training_indices, validation_indices = stratifier.split(
            dataset_y,
            test_set_fraction=1 - self.train_ratio,
            split_index=self.split_index_inner
        )
        return {
            "training_set_x": dataset_x.subset(training_indices),
            "training_set_y": dataset_y.subset(training_indices),
            "validation_set_x": dataset_x.subset(validation_indices),
            "validation_set_y": dataset_y.subset(validation_indices),
            "test_set_x": dataset_x.subset(self.test_indices),
            "test_set_y": dataset_y.subset(self.test_indices)
        }

    def get_model(self) -> sp.AbstractBaseModel:
        if self.model_name == "randomforest":
            rf_max_depth = self.model_hyperparams["rf_max_depth"]
            if rf_max_depth == -1:
                rf_max_depth = None
            sp_model = SklearnRandomForestRegressor(
                base_module=RandomForestRegressor(
                    n_estimators=self.model_hyperparams["rf_num_estimators"],
                    max_depth=rf_max_depth,
                    random_state=self.seed)
            )
        elif self.model_name == "bayesian_mlp":
            super_base_module = torch_model.TorchModel(
                base_module=pytorch_models.BayesianMLP(
                    input_size=SingleCycleApplication.get_dataset_x(self.feature_set_name,
                                                                    self.cache_directory).get_shape()[0][-1],
                    hidden_size=self.dn_hidden_layer_size),
                loss=CustomLoss(),
                batch_size=64,
                num_epochs=100
            )
            sp_model = meta_models.PytorchMLPRegressorWithUncertainty(
                model=super_base_module,
                num_target_samples=100
            )
        else:
            raise NotImplementedError(f"{self.model_name} does not exist.")
        self.model = sp_model
        return sp_model

    def train_model(self, model: sp.AbstractBaseModel) -> Optional[sp.AbstractBaseModel]:
        model.fit(self.datasets.training_set_x,
                  self.datasets.training_set_y)
        self.model = model
        return model

    def get_hyperopt_parameter_ranges(self) -> Dict[AnyStr, Union[List, Tuple]]:
        """
        Get hyper-parameter optimization ranges.

        Returns:
            A dictionary with each item corresponding to a named hyper-parameter and its associated discrete
            (represented as a Tuple) or continuous (represented as a List[start, end]) value range.
        """
        model_hyperopt_parameter_ranges = self.get_model_hyperparameter_ranges()
        hyperopt_ranges = {}
        hyperopt_ranges.update(model_hyperopt_parameter_ranges)
        return hyperopt_ranges

    def get_model_hyperparameter_ranges(self) -> Dict[AnyStr, Union[AnyStr, int]]:
        """
        Get the hyper-parameter optimization ranges for the evaluated model.

        Returns:
            A dictionary with each item corresponding to a named hyper-parameter and its associated discrete
            (represented as a Tuple) or continuous (represented as a List[start, end]) value range.
        """
        prefixes = {"randomforest": "rf_", "bayesian_mlp": "dn_"}
        model_hyperopt_parameter_ranges = self.get_model().get_hyperopt_parameter_ranges()
        model_hyperopt_parameter_ranges = update_dictionary_keys_with_prefixes(
            model_hyperopt_parameter_ranges, prefixes[self.model_name]
        )
        return model_hyperopt_parameter_ranges


if __name__ == "__main__":
    app = sp.instantiate_from_command_line(SingleCycleApplication)
    results = app.run()



