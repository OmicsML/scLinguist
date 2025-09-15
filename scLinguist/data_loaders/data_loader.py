import json
import os
import pyarrow.parquet as pq
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
from torch.utils.data import DataLoader, Dataset
from typing import Union
from warnings import warn

from scipy.sparse import issparse, csc_matrix, csr_matrix
from anndata import AnnData
import torch

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", -1))


def max_min_normalization_with_nan(data):
    High = 1
    Low = 1e-8
    scale = High - Low

    min_vals = np.nanmin(data, axis=1)
    max_vals = np.nanmax(data, axis=1)

    data = data.astype(float)
    nan_mask = np.isnan(data)

    for i in range(data.shape[0]):
        if np.all(nan_mask[i]):
            continue 
        data[i] -= min_vals[i]
        if max_vals[i] != min_vals[i]:
            data[i] /= (max_vals[i] - min_vals[i])
        data[i] *= scale
        data[i] += Low

    return data


def find_parquet_files(root_folder, suffix=".parquet"):
    parquet_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(suffix):
                parquet_files.append(os.path.join(root, file))
    return parquet_files


class spRNADataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = find_parquet_files(data_dir)
        print(self.files)
        self.files.sort(reverse=True)
        self.data_array, self.len_array, self.length = self.load_all_data()

    def __len__(self):
        return self.length

    def get_reverse_vocab(self):
        with open(self.reverse_vocab_path, "r") as f:
            vocab = json.load(f)
        return {int(k): v for k, v in vocab.items()}

    def get_origin_vocab(self):
        with open(self.origin_vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def get_row(self, file_index):
        group_id = file_index / 1000
        row_id = file_index % 1000
        return int(group_id), int(row_id)

    def load_all_data(self):
        data_list = []
        len_list = []
        length = 0

        for path in self.files:
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)
        return data_array, len_array, length

    def __getitem__(self, index):
        for i in range(len(self.len_array)):
            if index < self.len_array[i]:
                pdata = pq.ParquetFile(self.data_array[i])
                file_index = index if i == 0 else index - self.len_array[i - 1]
                break
        groups, rows = self.get_row(file_index)
        pdata = pdata.read_row_groups([groups])

        pgenes = pdata['genes'][rows].as_py()
        pexps = pdata['expressions'][rows].as_py()

        data = np.zeros(19202)
        data[pgenes] = pexps

        data = data.reshape(1, -1)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.squeeze()

        return data


class scRNADataset(Dataset):
    def __init__(self, data_dir, gene_order_path):
        self.gene_order_path = gene_order_path
        self.data_dir = data_dir
        # self.files = [f for f in list_files_in_dir(data_dir) if f.endswith(".h5ad")]
        gene_order = pd.read_csv(self.gene_order_path, index_col=0)
        self.gene_order = gene_order["0"]
        (
            self.data_array,
            self.len_array,
            self.length,
        ) = self.load_all_data()

    def __len__(self):
        return self.length

    def load_all_data(self):
        data_list = []
        sorted_data_list = []
        feature_id_list = []
        len_list = []
        length = 0

        # for path in self.files:
        for path in [self.data_dir]:
            adata = sc.read(path)
            adata.var_names = adata.var.feature_id
            adata = adata[:, self.gene_order]
            # tmp = hierarchical_bayesian_downsampling_csr(adata.X)
            # adata = anndata.AnnData(tmp)

            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata)
            data = adata.X
            # data = max_min_normalization(adata.X)
            # sorted_data, feature_id = sort(data.todense())
            # order and select gene
            data_list.append(data)
            # sorted_data_list.append(sorted_data)
            # feature_id_list.append(feature_id)

            length += adata.shape[0]
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)
        # sorted_data_array = np.array(sorted_data_list)
        # feature_id_array = np.array(feature_id_list)

        return data_array, len_array, length

    def __getitem__(self, index):
        data = self.data_array[0][index]
        data = torch.tensor(data.todense(), dtype=torch.float32)
        data = data.squeeze()
        return data



class scMultiDataset(Dataset):
    def __init__(self, data_dir_1=None, data_dir_2=None, adata_1=None, adata_2=None):
        """
        Initialize dataset from either file paths or AnnData objects.
        At least one of (data_dir_1, adata_1) and one of (data_dir_2, adata_2) must be provided.
        """
        assert (data_dir_1 is not None or adata_1 is not None), "Provide data_dir_1 or adata_1"
        assert (data_dir_2 is not None or adata_2 is not None), "Provide data_dir_2 or adata_2"
        # Load adata objects
        if adata_1 is None:
            self.adata_1 = sc.read(data_dir_1)
        else:
            self.adata_1 = adata_1.copy()

        if adata_2 is None:
            self.adata_2 = sc.read(data_dir_2)
        else:
            self.adata_2 = adata_2.copy()
        (
            self.data_array,
            self.len_array,
            self.data_array_1,
            self.len_array_1,
            self.length,
        ) = self.load_all_data()

    def __len__(self):
        return self.length

    def load_all_data(self):
        data_list = []
        len_list = []
        length = 0
        sc.pp.normalize_total(
            self.adata_1,
            target_sum=10000
        )
        sc.pp.log1p(self.adata_1)
        data = self.adata_1.X
        data_list.append(data)

        length += self.adata_1.shape[0]
        len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)

        data_list = []
        mask_list = []
        len_list = []
        length = 0


        data = self.adata_2.X.todense()

        data = max_min_normalization_with_nan(data)
        data_list.append(data)

        length += self.adata_2.shape[0]
        len_list.append(length)
        data_array_1 = np.array(data_list)
        len_array_1 = np.array(len_list)

        return data_array, len_array, data_array_1, len_array_1, length

    def __getitem__(self, index):
        data = self.data_array[0][index]
        data = torch.tensor(data.todense(), dtype=torch.float32)
        data = data.squeeze()

        data_1 = self.data_array_1[0][index]
        data_1 = data_1
        mask_idx = torch.tensor(~np.isnan(data_1))
        mask_idx = mask_idx.squeeze()
        data_1 = torch.tensor(np.nan_to_num(data_1, nan=1e-8), dtype=torch.float32)
        data_1 = data_1.squeeze()

        # return data_1, mask_idx, 1
        return data, data_1, mask_idx


def clr(adata: AnnData, inplace: bool = True, axis: int = 0) -> Union[None, AnnData]:
    """
    Apply the centered log ratio (CLR) transformation
    to normalize counts in adata.X.

    Args:
        data: AnnData object with protein expression counts.
        inplace: Whether to update adata.X inplace.
        axis: Axis across which CLR is performed.
    """

    if axis not in [0, 1]:
        raise ValueError("Invalid value for `axis` provided. Admissible options are `0` and `1`.")

    if not inplace:
        adata = adata.copy()

    if issparse(adata.X) and axis == 0 and not isinstance(adata.X, csc_matrix):
        warn("adata.X is sparse but not in CSC format. Converting to CSC.")
        x = csc_matrix(adata.X)
    elif issparse(adata.X) and axis == 1 and not isinstance(adata.X, csr_matrix):
        warn("adata.X is sparse but not in CSR format. Converting to CSR.")
        x = csr_matrix(adata.X)
    else:
        x = adata.X

    if issparse(x):
        print(list(np.log1p(x).sum(axis=axis).A))
        x.data /= np.repeat(
            np.exp(np.log1p(x).sum(axis=axis).A / x.shape[axis]), x.getnnz(axis=axis)
        )
        np.log1p(x.data, out=x.data)
    else:
        np.log1p(
            x / np.exp(np.log1p(x).sum(axis=axis, keepdims=True) / x.shape[axis]),
            out=x,
        )

    adata.X = x

    return None if inplace else adata


def list_files_in_dir(dir_path):
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def normalization(x, low=1e-8, high=1):
    MIN = min(x)
    MAX = max(x)
    x = low + (x - MIN) / (MAX - MIN) * (high - low)  # zoom to (low, high)
    return x


def fetch_files_in_subdirs(dir_path, suffix='.scb'):
    subdirs = []
    for _ in os.listdir(dir_path):
        cur_path = os.path.join(dir_path, _)
        if os.path.isdir(cur_path) and cur_path.endswith(suffix):
            subdirs.append(cur_path)

    file_paths = []
    for subdir in subdirs:
        file_paths.extend(list_files_in_dir(subdir))
    return file_paths


def sort(data):
    sorted_indices = np.argsort(-data, axis=1)
    sorted_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        row_indices = sorted_indices[i]
        sorted_data[i, :] = data[i, row_indices]
    return sorted_data, sorted_indices


class paProteinDataset(Dataset):
    def __init__(self, data_dir, mask_dir='./10M_mask/',
                 reverse_vocab_path='./tokenizer/reverse_vocab.json',
                 origin_vocab_path='./tokenizer/vocab.json',
                 protein_count=6428):
        self.data_dir = data_dir
        self.protein_count = protein_count
        self.reverse_vocab_path = reverse_vocab_path
        self.origin_vocab_path = origin_vocab_path

        self.files = [sd for sd in fetch_files_in_subdirs(data_dir) if sd.endswith(".parquet")]
        self.files.sort(reverse=False)
        self.masks = []
        for fs in self.files:
            last = fs.split('/')[-2]
            num = last.split('_')[1]
            self.masks.append(mask_dir + 'mask_' + str(num) + '_no_zero.json')
        (
            self.data_array,
            self.len_array,
            self.length,
        ) = self.load_all_data()
        self.fullmasks = self.load_mask_data()

    def __len__(self):
        return self.length

    def get_reverse_vocab(self):
        with open(self.reverse_vocab_path, "r") as f:
            vocab = json.load(f)
        return {int(k): v for k, v in vocab.items()}

    def get_origin_vocab(self):
        with open(self.origin_vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def get_row(self, file_index):
        group_id = file_index / 1000
        row_id = file_index % 1000
        return int(group_id), int(row_id)

    def load_all_data(self):
        data_list = []
        len_list = []
        length = 0

        for path in self.files:
            # print(path)
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)
        return data_array, len_array, length

    def load_mask_data(self):
        data_list = []

        # for path in self.files:
        for path in self.masks:
            # print(path)
            with open(path, "r") as f:
                mask = json.load(f)
            dataMatrix = np.array(list(mask.values()), dtype=object)
            data_list.append(dataMatrix)

        data_arrayMASK = np.array(data_list, dtype=object)

        return data_arrayMASK

    def __getitem__(self, index):
        self.vocab = self.get_reverse_vocab()
        self.gene_vocab = self.get_origin_vocab()

        for i in range(len(self.len_array)):
            if index < self.len_array[i]:
                pdata = pq.ParquetFile(self.data_array[i])
                fullmask = self.fullmasks[i]
                file_index = index if i == 0 else index - self.len_array[i - 1]
                break
        # Calculate the current row group number and the specific row within the group based on the random index
        
        groups, rows = self.get_row(file_index)
        pdata = pdata.read_row_groups([groups])
        # Perform column indexing on a row group to obtain the corresponding information 
        # The as_py() method is used to convert data types to basic Python data types
        pgenes = pdata['genes'][rows].as_py()
        pexps = pdata['expressions'][rows].as_py()
        # Create a data vector to store the expression matrix
        data = np.zeros(self.protein_count)

        # Use the genes array as an index array to index data, where the corresponding positions have expression values
        data[pgenes] = pexps

        data = np.nan_to_num(data)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.squeeze()
        tech = data[-1]
        data = data[:-1]

        mask_position = np.array(fullmask[file_index][:-1], dtype=int)
        mask = np.zeros(len(data))
        mask[mask_position] = 1
        mask = torch.tensor(mask, dtype=torch.int)
        mask = mask.squeeze()

        indices = torch.flatten(torch.nonzero(mask, as_tuple=False))
        assert len(mask_position) == len(indices)
        data = normalization(data)
        return (data, mask)


class paTESTProteinDataset_cytof(Dataset):
    def __init__(self, data_dir,
                 reverse_vocab_path='./tokenizer/reverse_vocab.json',
                 origin_vocab_path='./tokenizer/vocab.json',
                 mask_test_path='./mask_test_cytof.npy', protein_count=6428):
        self.data_dir = data_dir
        self.protein_count = protein_count
        self.reverse_vocab_path = reverse_vocab_path
        self.origin_vocab_path = origin_vocab_path
        self.files = [sd for sd in fetch_files_in_subdirs(data_dir) if sd.endswith(".parquet")]
        self.files.sort(reverse=False)
        self.mask = np.load(mask_test_path)
        self.mask = torch.tensor(self.mask, dtype=torch.int)
        self.mask = self.mask.squeeze()
        (
            self.data_array,
            self.len_array,
            self.length,
        ) = self.load_all_data()

    def __len__(self):
        return self.length

    def get_reverse_vocab(self):
        with open(self.reverse_vocab_path, "r") as f:
            vocab = json.load(f)
        return {int(k): v for k, v in vocab.items()}

    def get_origin_vocab(self):
        with open(self.origin_vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def get_row(self, file_index):
        group_id = file_index / 1000
        row_id = file_index % 1000
        return int(group_id), int(row_id)

    def load_all_data(self):
        data_list = []
        self.vocab = self.get_reverse_vocab()
        self.gene_vocab = self.get_origin_vocab()
        len_list = []
        length = 0

        for path in self.files:
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)
        return data_array, len_array, length

    def __getitem__(self, index):
        for i in range(len(self.len_array)):
            if index < self.len_array[i]:
                pdata = pq.ParquetFile(self.data_array[i])
                # fullmask = self.fullmasks[i]
                file_index = index if i == 0 else index - self.len_array[i - 1]
                break
        # Calculate the current row group number and the specific row within the group based on the random index
        groups, rows = self.get_row(file_index)
        pdata = pdata.read_row_groups([groups])
        # Perform column indexing on a row group to obtain the corresponding information (as_py() method is used to convert data types to basic Python data types)
        pgenes = pdata['genes'][rows].as_py()
        pexps = pdata['expressions'][rows].as_py()
        # Create a data vector to store the expression matrix
        data = np.zeros(self.protein_count)
        # Use the genes array as an index array to index data, where the corresponding positions have expression values
        data[pgenes] = pexps
        # nan2zero
        data = np.nan_to_num(data)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.squeeze()
        tech = data[-1]
        data = data[:-1]

        mask_position = np.array(self.mask[:-1])
        mask = np.zeros(len(data))
        mask[mask_position] = 1
        mask = torch.tensor(mask, dtype=torch.int)
        mask = mask.squeeze()

        indices = torch.flatten(torch.nonzero(mask, as_tuple=False))
        assert len(mask_position) == len(indices)
        data = normalization(data)
        return (data, mask)


class paTESTProteinDataset_citeseq(Dataset):
    def __init__(self, data_dir,
                 reverse_vocab_path='./tokenizer/reverse_vocab.json',
                 origin_vocab_path='./tokenizer/vocab.json',
                 mask_test_path='./mask_test_citeseq.npy', protein_count=6428):
        self.data_dir = data_dir
        self.protein_count = protein_count
        self.reverse_vocab_path = reverse_vocab_path
        self.origin_vocab_path = origin_vocab_path
        self.files = [sd for sd in fetch_files_in_subdirs(data_dir) if sd.endswith(".parquet")]
        self.files.sort(reverse=False)
        self.mask = np.load(mask_test_path)
        self.mask = torch.tensor(self.mask, dtype=torch.int)
        self.mask = self.mask.squeeze()
        (
            self.data_array,
            self.len_array,
            self.length,
        ) = self.load_all_data()

    def __len__(self):
        return self.length

    def get_reverse_vocab(self):
        with open(self.reverse_vocab_path, "r") as f:
            vocab = json.load(f)
        return {int(k): v for k, v in vocab.items()}

    def get_origin_vocab(self):
        with open(self.origin_vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def get_row(self, file_index):
        group_id = file_index / 1000
        row_id = file_index % 1000
        return int(group_id), int(row_id)

    def load_all_data(self):
        data_list = []
        self.vocab = self.get_reverse_vocab()
        self.gene_vocab = self.get_origin_vocab()
        len_list = []
        length = 0

        for path in self.files:
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)
        return data_array, len_array, length

    def __getitem__(self, index):
        for i in range(len(self.len_array)):
            if index < self.len_array[i]:
                pdata = pq.ParquetFile(self.data_array[i])
                # fullmask = self.fullmasks[i]
                file_index = index if i == 0 else index - self.len_array[i - 1]
                break
        groups, rows = self.get_row(file_index)
        pdata = pdata.read_row_groups([groups])
        pgenes = pdata['genes'][rows].as_py()
        pexps = pdata['expressions'][rows].as_py()
        data = np.zeros(self.protein_count)
        data[pgenes] = pexps
        # nan2zero
        data = np.nan_to_num(data)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.squeeze()
        tech = data[-1]
        data = data[:-1]

        mask_position = np.array(self.mask[:-1])
        mask = np.zeros(len(data))
        mask[mask_position] = 1
        mask = torch.tensor(mask, dtype=torch.int)
        mask = mask.squeeze()

        indices = torch.flatten(torch.nonzero(mask, as_tuple=False))
        assert len(mask_position) == len(indices)

        if torch.isnan(data).any():
            print("hahaha")
            print(list(data))
        data = normalization(data)
        return (data, mask)


def dataloader_generator(root_path, gene_order_path, batch_size=64, num_workers=0):
    dataset = paProteinDataset(data_dir=root_path)
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return data_loader


class spMultiDataset(Dataset):
    def __init__(self, data_dir_1, data_dir_2, mask_dir):
        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        self.files_1 = find_parquet_files(data_dir_1)
        print(self.files_1)
        self.files_1.sort(reverse=True)

        self.files_2 = find_parquet_files(data_dir_2)
        print(self.files_2)
        self.files_2.sort(reverse=True)
        self.masks = []
        for fs in self.files_2:
            print(fs)
            last = fs.split("/")[-1]
            num = last.split("_")[1].split(".")[0]
            self.masks.append(mask_dir + "ADT_" + str(num) + ".json")

        (
            self.data_array,
            self.len_array,
            self.data_array_1,
            self.len_array_1,
            self.length,
        ) = self.load_all_data()
        self.fullmasks = self.load_mask_data()

    def __len__(self):
        return self.length

    def get_reverse_vocab(self):
        with open(self.reverse_vocab_path, "r") as f:
            vocab = json.load(f)
        return {int(k): v for k, v in vocab.items()}

    def get_origin_vocab(self):
        with open(self.origin_vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def get_row(self, file_index):
        group_id = file_index / 1000
        row_id = file_index % 1000
        return int(group_id), int(row_id)

    def load_all_data(self):
        data_list = []
        len_list = []
        length = 0

        for path in self.files_1:
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array = np.array(data_list)
        len_array = np.array(len_list)

        data_list = []
        len_list = []
        length = 0

        for path in self.files_2:
            pdata = pq.ParquetFile(path)
            data_list.append(path)

            length += pdata.scan_contents()
            len_list.append(length)
        data_array_1 = np.array(data_list)
        len_array_1 = np.array(len_list)

        return data_array, len_array, data_array_1, len_array_1, length

    def load_mask_data(self):
        data_list = []

        for path in self.masks:
            with open(path, "r") as f:
                mask = json.load(f)
            dataMatrix = np.array(list(mask.values()), dtype=object)
            data_list.append(dataMatrix)

        data_arrayMASK = np.array(data_list, dtype=object)
        return data_arrayMASK

    def parquet_getitem(self, index, len_array, data_array, data_count):
        for i in range(len(len_array)):
            if index < len_array[i]:
                pdata = pq.ParquetFile(data_array[i])
                file_index = index if i == 0 else index - len_array[i - 1]
                break
        groups, rows = self.get_row(file_index)
        pdata = pdata.read_row_groups([groups])

        pgenes = pdata["genes"][rows].as_py()
        pexps = pdata["expressions"][rows].as_py()

        data = np.zeros(data_count)
        data[pgenes] = pexps
        return data, file_index, i

    def __getitem__(self, index):
        data, _, _ = self.parquet_getitem(index, self.len_array, self.data_array, 19202)
        data = data.reshape(1, -1)
        data = torch.tensor(data, dtype=torch.float32)
        data = data.squeeze()

        data_1, file_index, i = self.parquet_getitem(index, self.len_array_1, self.data_array_1, 6427)
        data_1 = torch.tensor(data_1, dtype=torch.float32)
        data_1 = data_1.squeeze()

        fullmask = self.fullmasks[i]
        mask_position = np.array(fullmask[int(file_index)][:-1], dtype=int)
        mask = np.zeros(len(data_1))
        mask[mask_position] = 1
        mask = torch.tensor(mask, dtype=torch.int)
        mask = mask.squeeze()

        indices = torch.flatten(torch.nonzero(mask, as_tuple=False))
        assert len(mask_position) == len(indices)

        data_1 = normalization(data_1, low=1e-8, high=1)


        return data, data_1, mask



def merge_duplicate_features(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Merge duplicated features in `adata` by summing expression values 
    for the same feature_id across columns.
    """
    X_dense = adata.X.toarray()
    feature_ids = adata.var["feature_id"].values

    unique_ids, inverse_indices = np.unique(feature_ids, return_inverse=True)
    merged_X = np.zeros((X_dense.shape[0], len(unique_ids)))

    for i in range(X_dense.shape[0]):
        merged_X[i] = np.bincount(
            inverse_indices, weights=X_dense[i], minlength=len(unique_ids)
        )

    merged = anndata.AnnData(X=csr_matrix(merged_X))
    merged.var["feature_id"] = unique_ids
    merged.obs = adata.obs.copy()

    return merged


def sum_adata_by_feature_id(adata: anndata.AnnData) -> anndata.AnnData:
    """
    Sum expression values across duplicated feature_ids in columns.
    Keeps unique features as-is and merges duplicated ones.
    """
    # Remove features without a valid ID
    adata = adata[:, ~adata.var["feature_id"].isna()].copy()

    feature_ids = adata.var["feature_id"]
    unique_ids, inverse_indices, counts = np.unique(
        feature_ids, return_inverse=True, return_counts=True
    )

    # Keep non-duplicated features
    unique_mask = feature_ids.isin(unique_ids[counts == 1])
    adata_unique = adata[:, unique_mask]

    # Merge duplicated features
    dup_mask = feature_ids.isin(unique_ids[counts > 1])
    adata_dup = adata[:, dup_mask]
    adata_merged = merge_duplicate_features(adata_dup)

    # Combine both
    return anndata.concat([adata_unique, adata_merged], axis=1)


def align_and_merge_with_order(rna: anndata.AnnData, order: list) -> anndata.AnnData:
    """
    Aligns the input RNA AnnData object to a predefined gene order.

    This function:
    - Filters genes to keep only those in the target `order`
    - Adds missing genes as all-0 dummy values to preserve column alignment
    - Reorders columns to match the exact `order`
    """
    # Create a dummy row to enforce inclusion of all genes in `order`
    dummy = anndata.AnnData(X=np.zeros((1, len(order))))
    dummy.var_names = order
    dummy.obs_names = ["__dummy__"]

    # Keep only genes that appear in both `rna` and `order`
    valid_genes = list(set(order) & set(rna.var_names))
    rna = rna[:, valid_genes].copy()

    # Concatenate dummy row and real data to ensure all genes in `order` are present
    rna = sc.concat([dummy, rna], axis=0, join="outer")

    # Drop the dummy row and reorder columns according to `order`
    rna = rna[1:, dummy.var_names]

    return rna
