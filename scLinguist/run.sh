#!/bin/bash

### pretraining with RNA
python train.py --mode RNA --train_data_path data/RNA/3m_cell_databanks_log \
--test_data_path data/RNA/test_data/PBMC_3.h5ad --wandb_name RNA \
--enc_vocab_len 19202 --dec_vocab_len 19202 --mask_prob 0.6 \

### pretraining with protein
train.py --mode protein --train_data_path /public/home/hpc214701031/project/scTrans/data/RNA/3m_cell_databanks_log \
--test_data_path /public/home/hpc214701031/project/scTrans/data/RNA/test_data/PBMC_3.h5ad --wandb_name protein \
--enc_vocab_len 6427 --dec_vocab_len 6427 \


### post-pretraining with paired-data
train.py --mode RNA-protein --train_data_path /public/home/hpc214701031/project/scTrans/data/RNA/3m_cell_databanks_log \
--train_data_path_1 /public/home/hpc214701031/project/scTrans/data/protein/3m_cell_databanks_log \
--mask_data_path /public/home/hpc214701031/project/scTrans/data/RNA/3m_cell_databanks_log \
--test_data_path_1 /public/home/hpc214701031/project/scTrans/data/RNA/test_data/PBMC_3.h5ad \
--test_data_path /public/home/hpc214701031/project/scTrans/data/RNA/test_data/PBMC_3.h5ad --wandb_name RNA-protein \
--enc_vocab_len 19202 --dec_vocab_len 6427 \