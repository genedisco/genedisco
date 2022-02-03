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


class CCLEProteinQuantification(object):
    """
    SOURCE: https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz
    PROJECT: https://gygi.med.harvard.edu/publications/ccle

    CITE:
    David P. Nusinow, John Szpyt, Mahmoud Ghandi, Christopher M. Rose, E. Robert McDonald III, Marian Kalocsay,
    Judit Jané-Valbuena, Ellen Gelfand, Devin K. Schweppe, Mark Jedrychowski, Javad Golji, Dale A. Porter,
    Tomas Rejtar, Y. Karen Wang, Gregory V. Kryukov, Frank Stegmeier, Brian K. Erickson, Levi A. Garraway,
     William R. Sellers, Steven P. Gygi (2020). Quantitative Proteomics of the Cancer Cell Line Encyclopedia.
     Cell 180, 2. https://doi.org/10.1016/j.cell.2019.12.02

    ACKNOWLEDGE:
    You agree to acknowledge in any work based in whole or part on the Data,
    the published paper from which the Data derives, the version of the Data, and the role of the Broad in
    its distribution. You agree to use the acknowledgement wording provided for the relevant Data in its publication.
    You will also declare in any such work that those who carried out the original analysis and collection of the Data
    bear no responsibility for the further analysis or interpretation of it.

    LICENSE: https://portals.broadinstitute.org/ccle/about
    """
    FILE_URL = "https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz"

    @staticmethod
    def load_data(save_directory) -> AbstractDataSource:
        h5_file = os.path.join(save_directory, "ccle_protein_quantification.h5")
        if not os.path.exists(h5_file):
            gz_file_path = os.path.join(save_directory, "ccle_protein_quantification.csv.gz")
            if not os.path.exists(gz_file_path):
                sp.download_streamed(CCLEProteinQuantification.FILE_URL, gz_file_path)
            df = pd.read_csv(gz_file_path, compression='gzip',
                             index_col="Gene_Symbol")
            excluded_columns = ["Protein_Id", "Description", "Group_ID", "Uniprot", "Uniprot_Acc"]
            excluded_indices = [df.columns.values.tolist().index(name) for name in excluded_columns]
            included_indices = list(sorted(set(list(range(len(df.columns)))) - set(excluded_indices)))
            data = df.values[:, included_indices].astype(float)

            si = SimpleImputer(missing_values=float("nan"), strategy='mean')
            data = si.fit_transform(data)

            row_names = df.index.values.tolist()
            name_missing_indices = np.where(list(map(lambda x: isinstance(x, float) and np.isnan(x), row_names)))[0]
            for idx in name_missing_indices:
                row_names[idx] = ""

            name_converter = HGNCNames(save_directory)
            row_names = name_converter.update_outdated_gene_names(row_names)
            row_names, idx_start = np.unique(sorted(row_names), return_index=True)
            data = data[idx_start]
            col_names = df.columns.values[included_indices].tolist()

            HDF5Tools.save_h5_file(h5_file,
                                   data,
                                   "ccle_protein_quantification",
                                   column_names=col_names,
                                   row_names=row_names)
        data_source = HDF5DataSource(h5_file, fill_missing_value=0,
                                     duplicate_merge_strategy=sp.FirstEntryMergeStrategy())
        return data_source
