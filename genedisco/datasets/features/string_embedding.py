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


class STRINGEmbedding(object):
    """
    SOURCE:
    https://groups.csail.mit.edu/cb/mashup/vectors/string_human_genes.txt
    https://groups.csail.mit.edu/cb/mashup/vectors/string_human_mashup_vectors_d800.txt

    PROJECT: http://cb.csail.mit.edu/cb/mashup/

    CITE:
    https://string-db.org/cgi/about?sessionId=bGpYHQrYvsOz&footer_active_subpage=references

    ACKNOWLEDGE:
    https://string-db.org/cgi/about?sessionId=bGpYHQrYvsOz&footer_active_subpage=references

    LICENSE: https://creativecommons.org/licenses/by/4.0/
    """
    FILE_URL = "https://groups.csail.mit.edu/cb/mashup/vectors/string_human_mashup_vectors_d800.txt"
    NAME_FILE_URL = "https://groups.csail.mit.edu/cb/mashup/vectors/string_human_genes.txt"

    @staticmethod
    def load_data(save_directory) -> AbstractDataSource:
        h5_file = os.path.join(save_directory, "string_embedding.h5")
        if not os.path.exists(h5_file):
            csv_file_path = os.path.join(save_directory, "string_human_mashup_vectors_d800.txt")
            if not os.path.exists(csv_file_path):
                sp.download_streamed(STRINGEmbedding.FILE_URL, csv_file_path)
            name_file_path = os.path.join(save_directory, "string_human_genes.txt")
            if not os.path.exists(name_file_path):
                sp.download_streamed(STRINGEmbedding.NAME_FILE_URL, name_file_path)

            df = pd.read_csv(csv_file_path, index_col=None, header=None, sep="\t")
            row_names = pd.read_csv(name_file_path, index_col=None, header=None).values[:, 0].tolist()

            data = df.values[:, 1:]
            name_converter = HGNCNames(save_directory)
            row_names = name_converter.update_outdated_gene_names(row_names)
            row_names, idx_start = np.unique(sorted(row_names), return_index=True)
            data = data[idx_start]
            col_names = [f"feature_{idx}" for idx in range(799)]

            HDF5Tools.save_h5_file(h5_file,
                                   data,
                                   "string_embedding",
                                   column_names=col_names,
                                   row_names=row_names)
        data_source = HDF5DataSource(h5_file, fill_missing_value=0,
                                     duplicate_merge_strategy=sp.FirstEntryMergeStrategy())
        return data_source
