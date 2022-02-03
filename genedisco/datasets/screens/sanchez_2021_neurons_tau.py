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
import pandas as pd
import slingpy as sp
from genedisco.datasets.features.hgnc_names import HGNCNames
from slingpy.data_access.data_sources.hdf5_tools import HDF5Tools
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource


class Sanchez2021NeuronsTau(object):
    """
    Data from: Genome-wide CRISPR screen identifies protein pathways modulating tau protein levels in neurons.
    Communications Biology 2021
    https://www.nature.com/articles/s42003-021-02272-1#MOESM4

    LICENSE: https://creativecommons.org/licenses/by/4.0/
    """
    @staticmethod
    def load_data(save_directory) -> AbstractDataSource:
        h5_file = os.path.join(save_directory, "sanchez_2021_neurons_tau.h5")
        if not os.path.exists(h5_file):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            csv_file_path = os.path.join(dir_path, "sanchezetal2021_21days.csv")
            df = pd.read_csv(csv_file_path, sep=",", index_col="Symbol")

            gene_names, data = df.index.values.tolist(), df[['RSA_Down']].values.astype(np.float32)

            name_converter = HGNCNames(save_directory)
            gene_names = name_converter.update_outdated_gene_names(gene_names)
            gene_names, idx_start = np.unique(sorted(gene_names), return_index=True)
            data = data[idx_start]

            HDF5Tools.save_h5_file(h5_file,
                                   -data,
                                   "sanchez_2021_neurons_tau",
                                   column_names=["RSA"],
                                   row_names=gene_names)
        data_source = HDF5DataSource(h5_file, duplicate_merge_strategy=sp.MeanMergeStrategy())
        return data_source
