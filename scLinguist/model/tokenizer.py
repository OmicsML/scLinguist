import torch
import numpy as np
from torch.distributions import Bernoulli, Binomial, Beta

def hierarchical_bayesian_downsampling(X, threshold=1000):
    # X is a n*p tensor representing the gene expression matrix
    n, p = X.shape

    # Get the current device of X
    current_device = X.device

    # First hierarchy: For each cell, decide whether to downsample based on total expression
    total_expression = X.sum(dim=1)
    # Cells with total expression below the threshold are not downsampled (gamma = 0)
    gamma = torch.where(total_expression < threshold, torch.zeros(n, device=current_device), Bernoulli(torch.ones(n, device=current_device) * 0.5).sample())
    # print(gamma)

    # Second hierarchy: Binomial downsampling for cells where gamma = 1
    b = Beta(2, 2).sample((n, 1)).to(current_device)  # Downsampling rate for each cell

    # Apply downsampling using Binomial distribution
    downsampled = Binomial(X, b).sample()

    # Apply gamma to select between original and downsampled values
    X_input = torch.where(gamma.unsqueeze(1) == 1, downsampled, X)
    X_input = X_input.to(current_device)


    return X_input


# def mask_data(tensor, mask_probability):
#     mask_matrix = torch.rand_like(tensor) < mask_probability
#     masked_tensor = torch.where(mask_matrix, torch.full_like(tensor, -1), tensor)

#     return masked_tensor, mask_matrix

def mask_data_RNA(tensor, mask_probability):
    # Create a mask for non-zero values with the given mask_probability
    mask_non_zero = torch.rand_like(tensor) < mask_probability
    # Create a mask for zero values with 1/10 of the mask_probability
    mask_zero = torch.rand_like(tensor) < (mask_probability / 10.0)

    # Combine masks so that the final mask is True if either condition is met
    mask_matrix = torch.where(tensor != 0, mask_non_zero, mask_zero)
    # Apply the mask to the tensor
    masked_tensor = torch.where(mask_matrix, torch.full_like(tensor, -1), tensor)

    return masked_tensor, mask_matrix

def mask_data_ADT(tensors, mask_probability):
    _data, mask = tensors
    min_data = min(_data[mask.type(torch.bool)])
    # Create a mask for non-zero values with the given mask_probability
    mask_non_zero = torch.rand_like(_data) < mask_probability
    # Create a mask for zero values with 1/10 of the mask_probability
    mask_zero = torch.rand_like(_data) < mask_probability
    intersection_tensor = (_data != min_data) & mask.type(torch.bool)
    # Combine masks so that the final mask is True if either condition is met
    mask_matrix = torch.where(intersection_tensor, mask_non_zero, mask_zero)
    # Apply the mask to the tensor
    intersection_tensor = mask_matrix & mask.type(torch.bool)
    masked_tensor = torch.where(intersection_tensor, torch.full_like(_data, 0), _data)

    return masked_tensor, mask, intersection_tensor


def data_tokenizer(gene_ids, datas, zero_gene=True):
    """
    Parameters:
        gene_ids: (batch_size, n_features)
            a batch of the id of data's gene
        datas: (batch_size, n_features)
            a batch of cell x gene expression value data
        zero_gene: bool
            whether include zero_gene
    Return:
        tokenized_data: list[ tuple(gene_id, count) ]
             data after tokenizer
    """
    assert datas.shape[1] == gene_ids.shape[1]

    tokenized_data = []
    for i in range(len(datas)):
        row_data = datas[i]
        row_gene = gene_ids[i]
        if zero_gene:
            value = row_data
            gene = row_gene
        else:
            idx = torch.nonzero((row_data != 0) & (row_data != -1)).squeeze()
            value = row_data[idx]
            gene = row_gene[idx]

        tokenized_data.append((gene, value))

    return tokenized_data


def data_padding(gene_len, datas, padding_id, padding_value=0):
    """
    Parameters:
        datas: list of [gene, value]
            a batch of cell x gene expression value data after tokenizer
        gene_len: int
            the number of all gene (include non gene)
        padding_id: int
            the padding token id
        padding_value: int
            the padding token value

    Return:
        tokenized_data: Dict[str, torch.Tensor]
             data after padding
    """
    # Pre-allocate tensors with padding_id and padding_value
    gene_ids = torch.full((len(datas), gene_len), padding_id, dtype=datas[0][0].dtype, device=datas[0][0].device)
    values = torch.full((len(datas), gene_len), padding_value, dtype=datas[0][1].dtype, device=datas[0][1].device)

    for i, (gene_id, value) in enumerate(datas):
        # Determine the length to copy based on the gene_len
        length_to_copy = min(gene_id.shape[0], gene_len)
        
        if gene_id.shape[0] <= gene_len:
            # Copy the data directly to the pre-allocated tensors
            gene_ids[i, :length_to_copy] = gene_id[:length_to_copy]
            values[i, :length_to_copy] = value[:length_to_copy]
        else:
            # Randomly sample indices and copy the sampled data
            indices = torch.randperm(gene_id.shape[0])[:gene_len]
            gene_ids[i] = gene_id[indices]
            values[i] = value[indices]

    return gene_ids, values


def data_tokenizer_padding(
    datas, gene_ids, zero_gene, padding_id, padding_value, gene_len
):
    """
    Parameters:
        gene_ids: (batch_size, n_features)
            a batch of the id of data's gene
        datas: (batch_size, n_features)
            a batch of cell x gene expression value data
        zero_gene: bool
            whether include zero_gene
        padding_id: int
            the padding token id
        padding_value: int
            the padding token value

    Return:
        tokenized_data: Dict[str, torch.Tensor]
             data after padding
    """

    tokenized_data = data_tokenizer(gene_ids=gene_ids, datas=datas, zero_gene=zero_gene)
    genes, values = data_padding(
        datas=tokenized_data,
        gene_len=gene_len,
        padding_id=padding_id,
        padding_value=padding_value,
    )

    return genes, values
