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
from sklearn.impute import SimpleImputer
from genedisco.datasets.features.hgnc_names import HGNCNames
from slingpy.data_access.data_sources.hdf5_tools import HDF5Tools
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource


class Achilles(object):
    """
    Data from:
    SOURCE: https://ndownloader.figshare.com/files/25494359
    PROJECT: https://depmap.org/portal/download/

    CITE:
    DepMap, Broad (2020): DepMap 20Q4 Public. figshare. Dataset doi:10.6084/m9.figshare.13237076.v1.
    Robin M. Meyers, Jordan G. Bryan, James M. McFarland, Barbara A. Weir, ... David E. Root, William C. Hahn,
    Aviad Tsherniak. Computational correction of copy number effect improves specificity of CRISPR-Cas9 essentiality
    screens in cancer cells. Nature Genetics 2017 October 49:1779–1784. doi:10.1038/ng.3984
    Dempster, J. M., Rossen, J., Kazachkova, M., Pan, J., Kugener, G., Root, D. E., & Tsherniak, A. (2019).
    Extracting Biological Insights from the Project Achilles Genome-Scale CRISPR Screens in Cancer Cell Lines.
    BioRxiv, 720243.

    ACKNOWLEDGE:
    This project is partially funded by CTD2, the Achilles consortium, and The Carlos Slim Foundation in
    Mexico through the Slim Initiative for Genomic Medicine.

    LICENSE: https://creativecommons.org/licenses/by/4.0/
    """
    FILE_URL = "https://ndownloader.figshare.com/files/25494359"

    @staticmethod
    def load_data(save_directory) -> AbstractDataSource:
        h5_file = os.path.join(save_directory, "achilles.h5")
        if not os.path.exists(h5_file):
            csv_file_path = os.path.join(save_directory, "achilles.csv")
            if not os.path.exists(csv_file_path):
                sp.download_streamed(Achilles.FILE_URL, csv_file_path)
            df = pd.read_csv(csv_file_path)
            gene_names = list(map(lambda x: x.split(" ")[0], df.columns.values.tolist()[1:]))
            data = df.values[:, 1:].astype(float).transpose()

            si = SimpleImputer(missing_values=float("nan"), strategy='mean')
            data = si.fit_transform(data)

            name_converter = HGNCNames(save_directory)
            gene_names = name_converter.update_outdated_gene_names(gene_names)
            gene_names, idx_start = np.unique(sorted(gene_names), return_index=True)
            data = data[idx_start]
            HDF5Tools.save_h5_file(h5_file,
                                   data,
                                   "achilles",
                                   column_names=df["DepMap_ID"].values.tolist(),
                                   row_names=gene_names)
        data_source = HDF5DataSource(h5_file, fill_missing_value=0)
        return data_source
