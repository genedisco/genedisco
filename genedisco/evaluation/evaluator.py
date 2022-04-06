"""
Copyright (C) 2022  Arash Mehrjou, GlaxoSmithKline plc

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import os
import pickle
import numpy as np
from typing import AnyStr, Optional, List
from collections import OrderedDict
from slingpy.utils.logging import info, warn
from slingpy.utils.path_tools import PathTools
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy.evaluation.metrics.abstract_metric import AbstractMetric
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource
from slingpy.data_access.data_sources.composite_data_source import CompositeDataSource
from genedisco.datasets.features.achilles import Achilles
from genedisco.datasets.features.string_embedding import STRINGEmbedding
from genedisco.datasets.screens.zhuang_2019_nk_cancer import Zhuang2019NKCancer
from genedisco.datasets.screens.schmidt_2021_t_cells_il2 import Schmidt2021TCellsIL2
from genedisco.datasets.screens.sanchez_2021_neurons_tau import Sanchez2021NeuronsTau
from genedisco.datasets.screens.schmidt_2021_t_cells_ifng import Schmidt2021TCellsIFNg
from genedisco.datasets.features.ccle_protein_quantification import CCLEProteinQuantification
from genedisco.datasets.screens.zhu_2021_sarscov2_host_factors import Zhu2021SARSCoV2HostFactors


class Evaluator(object):
    
    @staticmethod
    def evaluate(top_movers_filepath:AnyStr, 
                 super_dir_to_cycle_dirs: AnyStr, 
                 metrics: List[AbstractMetric],
                 with_print=False):
        
        metric_dict = OrderedDict()
        for metric in metrics:
                metric_name = metric.__class__.__name__
                value = metric.evaluate(top_movers_filepath, super_dir_to_cycle_dirs)
                if metric_name in metric_dict:
                    message = f"{metric_name} was already present in metric dict. " \
                            f"Do you have two metrics set for evaluation with the same name?"
                    warn(message)
                    raise AssertionError(message)
                metric_dict[metric_name] = value
        if with_print:
            info(f"HitRatio Performance on", metric_dict)
        return metric_dict
    
    
def save_top_movers(top_ratio_threshold:Optional[float], 
                    feature_set_name: AnyStr, 
                    dataset_name: AnyStr,
                    cache_directory: AnyStr,
                    test_ratio: float,
                    seed: int,
                    dir_to_save: AnyStr):
    """Save the top mover genes of the proivded dataset for future use in the evaluation

    Args:
        top_ratio_threshold (float, optional): The ratio from the head of the sorted list of genes that are considered as top movers.
        feature_set_name (AnyStr): The name of the feature set used for active learning cycles.
        dataset_name (AnyStr): The name of the dataset used for active learning cycles.
        cache_directory (AnyStr): The path to dataset cache.
        test_ratio: What ratio of dataset samples are used as for test.
        seed: The random seed. It is used here to reproduce the dataset stratification that is done in the AL loop.
        dir_to_save (AnyStr): The  path to save the top mover genes.
    Returns:
        full_path_to_save: The full path to where the top mover file is saved. 
    """
    TOP_MOVER_FILENAME = "top_movers_seed_{}.pkl"
    np.random.seed(seed)
    dataset_x = get_dataset_x(feature_set_name, cache_directory)
    dataset_y = get_dataset_y(dataset_name, cache_directory)
    avail_names = sorted(list(set(dataset_x.get_row_names()).intersection(set(dataset_y.get_row_names()))))
    dataset_y = dataset_y.subset(avail_names)
    dataset_x = dataset_x.subset(avail_names)
    
    avail_indices = sorted(
    list(set(dataset_x.get_row_names()).intersection(set(dataset_y.get_row_names())))
    )
    test_indices = sorted(
        list(
            np.random.choice(
                avail_indices, 
                size=int(test_ratio * len(avail_indices)), 
                replace=False)
        )
    )
    training_indices = list((set(avail_names) - set(test_indices)))
    target_values = np.squeeze(dataset_y.subset(training_indices).get_data()[0])
    num_top_hits = int(len(target_values) * top_ratio_threshold)
    top_target_indices = np.flip(np.argsort(np.abs(np.squeeze(target_values))))[:num_top_hits]
    top_mover_indices = np.array(training_indices)[top_target_indices]
    PathTools.mkdir_if_not_exists(dir_to_save)
    full_path_to_save = os.path.join(dir_to_save, TOP_MOVER_FILENAME.format(seed))
    with open(full_path_to_save, "wb") as fp:
        pickle.dump(top_mover_indices, fp)
    return full_path_to_save

def get_dataset_x(feature_set_name, cache_directory): 
    # Load the feature set
    if feature_set_name == "achilles":
        dataset = Achilles.load_data(cache_directory)
    elif feature_set_name == "ccle":
        dataset = CCLEProteinQuantification.load_data(cache_directory)
    elif feature_set_name == "string":
        dataset = STRINGEmbedding.load_data(cache_directory)
    else:
        raise NotImplementedError(f"{feature_set_name} is not implemented.")
    dataset_x = CompositeDataSource([dataset])
    return dataset_x

def get_dataset_y(dataset_name, cache_directory):        
    # Load the target dataset
    if dataset_name == "schmidt_2021_ifng":
        dataset_y = Schmidt2021TCellsIFNg.load_data(cache_directory)
    elif dataset_name == "schmidt_2021_il2":
        dataset_y = Schmidt2021TCellsIL2.load_data(cache_directory)
    elif dataset_name == "zhuang_2019_nk":
        dataset_y = Zhuang2019NKCancer.load_data(cache_directory)
    elif dataset_name == "sanchez_2021_tau":
        dataset_y = Sanchez2021NeuronsTau.load_data(cache_directory)
    elif dataset_name == "zhu_2021_sarscov2":
        dataset_y = Zhu2021SARSCoV2HostFactors.load_data(cache_directory)
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented.")
    return dataset_y