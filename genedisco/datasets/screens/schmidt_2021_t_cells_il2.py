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


class Schmidt2021TCellsIL2(object):
    """
    Data from: CRISPR activation and interference screens in primary human T cells decode cytokine
    regulation. bioRxiv 2021
    https://www.biorxiv.org/content/10.1101/2021.05.11.443701v1
    GEOS URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174292

    Commands:
    mageck test -k GSE174255_reads_A.txt -t 32_DolcettoSetA_Donor1_IL2_high -c 31_DolcettoSetA_Donor1_IL2_low -n schmidt_il2_d1
    mageck test -k GSE174255_reads_A.txt -t 44_DolcettoSetA_Donor2_IL2_high -c 43_DolcettoSetA_Donor2_IL2_low -n schmidt_il2_d2

    LICENSE: https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html
    """
    @staticmethod
    def load_data(save_directory) -> AbstractDataSource:
        h5_file = os.path.join(save_directory, "schmidt_2021_il2.h5")
        if not os.path.exists(h5_file):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            d1_file_path = os.path.join(dir_path, "schmidt_il2_d1.gene_summary.txt")
            df = pd.read_csv(d1_file_path, sep="\t", index_col="id")
            d2_file_path = os.path.join(dir_path, "schmidt_il2_d2.gene_summary.txt")
            df_d2 = pd.read_csv(d2_file_path, sep="\t", index_col="id")

            df = pd.concat([df, df_d2])
            group_by_row_index = df.groupby(df.index)
            df = group_by_row_index.mean()

            gene_names, data = df.index.values.tolist(), df[['pos|lfc']].values.astype(np.float32)

            name_converter = HGNCNames(save_directory)
            gene_names = name_converter.update_outdated_gene_names(gene_names)
            gene_names, idx_start = np.unique(sorted(gene_names), return_index=True)
            data = data[idx_start]

            HDF5Tools.save_h5_file(h5_file,
                                   data,
                                   "schmidt_2021_il2",
                                   column_names=["log-fold-change"],
                                   row_names=gene_names)
        data_source = HDF5DataSource(h5_file, duplicate_merge_strategy=sp.MeanMergeStrategy())
        return data_source
