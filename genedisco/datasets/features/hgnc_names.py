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
from typing import AnyStr
from slingpy.utils.logging import warn


class HGNCNames(object):
    FILE_URL = "http://ftp.ebi.ac.uk/pub/databases/genenames/out_of_date_hgnc/tsv/locus_types/gene_with_protein_product.txt"

    def __init__(self, cache_directory: AnyStr):
        self.hgnc_mappings = {}
        self.cache_directory = cache_directory

    def get_hgnc_master_file(self):
        tsv_file = os.path.join(self.cache_directory, "hgnc_mapping.tsv")
        if not os.path.exists(tsv_file):
            sp.download_streamed(HGNCNames.FILE_URL, tsv_file)
        return tsv_file

    def get_gene_names(self, index_name: AnyStr = "symbol"):
        tsv_file = self.get_hgnc_master_file()
        data = pd.read_csv(tsv_file, sep="\t", index_col=index_name)
        gene_names = data.index.values.tolist()
        return gene_names

    def get_hgnc_mapping(self, from_id: AnyStr = "uniprot_ids", to_id: AnyStr = "symbol",
                         split_symbol: AnyStr = "|"):
        cache_entry_name = f"{from_id}${to_id}"
        if cache_entry_name not in self.hgnc_mappings:
            tsv_file = self.get_hgnc_master_file()
            mapping_data = pd.read_csv(tsv_file, sep="\t", index_col=from_id)
            target_column_index = mapping_data.columns.values.tolist().index(to_id)
            mapping_row_names = mapping_data.index.values
            mapping_data = mapping_data.values[:, target_column_index]

            if from_id in ["prev_symbol", "alias_symbol"]:
                new_mapping_data, new_mapping_row_names = [], []
                for idx, row_name in enumerate(mapping_row_names):
                    if isinstance(row_name, float) and np.isnan(row_name):
                        continue

                    split_names = row_name.split(split_symbol)
                    for split_name in split_names:
                        if split_name == "":
                            continue
                        new_mapping_data.append(mapping_data[idx])
                        new_mapping_row_names.append(split_name)
                mapping_data = np.array(new_mapping_data)
                mapping_row_names = new_mapping_row_names

            assert len(mapping_row_names) == len(mapping_data)

            circular_entries = []
            mapping = dict(zip(mapping_row_names, list(map(str, mapping_data))))
            for from_id, to_id in mapping.items():
                cur_to_id = to_id
                circle_detected, previous_stack = False, {cur_to_id}
                while cur_to_id in mapping:
                    cur_to_id = mapping[cur_to_id]
                    if cur_to_id in previous_stack:
                        circle_detected = True
                        break
                    else:
                        previous_stack.add(cur_to_id)
                if circle_detected:
                    circular_entries.append(to_id)
                elif cur_to_id != to_id:  # Replace with final node in graph.
                    mapping[from_id] = cur_to_id

            for entry in set(circular_entries):
                mapping[entry] = entry
            self.hgnc_mappings[cache_entry_name] = mapping
        return self.hgnc_mappings[cache_entry_name]

    def update_outdated_gene_names(self, row_names, verbose: bool = False):
        row_names = list(row_names)
        for other_id_name in ["prev_symbol", "alias_symbol"]:
            previous_mapping = self.get_hgnc_mapping(to_id="symbol", from_id=other_id_name)
            for idx, row_name in enumerate(row_names):
                before = row_name
                if row_name in previous_mapping:
                    row_names[idx] = previous_mapping[row_name]
                if before != row_names[idx] and verbose:
                    warn(f"{before} => {row_names[idx]}")
        return row_names
