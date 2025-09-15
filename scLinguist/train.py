import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loaders.data_loader import paProteinDataset, paTESTProteinDataset_citeseq, scRNADataset, spRNADataset, spMultiDataset, scMultiDataset
import argparse
from model.model import scTrans
from model.configuration_hyena import HyenaConfig

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", -1))
print("LOCAL RANK:", LOCAL_RANK)
print("RANK:", RANK)
print("WORLD_SIZE:", WORLD_SIZE)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./data/RNA/cell_gene",
        help="path to the training data for RNA or protein mode",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./data/RNA/cell_gene",
        help="path to the testing data for RNA or protein mode",
    )
    parser.add_argument(
        "--train_data_path_1",
        type=str,
        default="./data/RNA/cell_gene",
        help="path to the training data for RNA-protein mode",
    )
    parser.add_argument(
        "--test_data_path_1",
        type=str,
        default="./data/RNA/cell_gene",
        help="path to the testing data for RNA-protein mode",
    )
    parser.add_argument(
        "--mask_data_path",
        type=str,
        default="./data/RNA/cell_gene",
        help="path to the mask data for RNA-protein mode",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="RNA",
        help="mode: protein, RNA, or RNA-protein (default: RNA)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./result2024",
        help="path to save the model and results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument("--freeze_layers", action="store_true")
    parser.add_argument("--syncBN", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--start_epochs", default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--device", type=str, default=8, help="device = 'cpu' or '0' or '0,1,2,3'"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--mask_prob", type=float, default=0.3, help="mask probability")
    parser.add_argument("--hyena_dropout", type=float, default=0.0, help="hyena dropout")
    parser.add_argument(
        "--enc_vocab_len", type=int, default=6427, help="vocab length of encoder"
    )
    parser.add_argument(
        "--dec_vocab_len", type=int, default=6427, help="vocab length of decoder"
    )
    parser.add_argument(
        "--enc_depth", type=int, default=1, help="sequence length of decoder"
    )
    parser.add_argument(
        "--dec_depth", type=int, default=1, help="sequence length of decoder"
    )
    parser.add_argument(
        "--enc_dim", type=int, default=32, help="latent dimension of each token"
    )
    parser.add_argument(
        "--dec_dim", type=int, default=32, help="latent dimension of each token"
    )
    parser.add_argument(
        "--position_emb_dim",
        type=int,
        default=5,
        help="dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands",
    )
    parser.add_argument(
        "--encoder_checkpoint_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--decoder_checkpoint_path",
        type=str,
        default=None,
    )
    parser.add_argument("--wandb_name", type=str, default="test", help="wanbd name")

    args = parser.parse_args()
    config = vars(args)
    return config



def load_data(
        train_data_dir,
        test_data_dir,
        mode="protein", 
        train_data_dir_1=None,
        test_data_dir_1=None,
        gene_order_path=None,
        mask_dir=None
    ):
    if mode == "protein":
        # Create data
        all_data = paProteinDataset(
            train_data_dir=train_data_dir,
        )
        n_train = int(len(all_data) * 0.9)
        train_data, valid_data = torch.utils.data.random_split(
            all_data, [n_train, len(all_data) - n_train]
        )
        test_data = paTESTProteinDataset_citeseq(
            test_data_dir=test_data_dir,
        )
    elif mode == "RNA":
        all_data = spRNADataset(
            data_dir=train_data_dir
            )
        n_train = int(len(all_data) * 0.9)
        train_data, valid_data = torch.utils.data.random_split(
            all_data, [n_train, len(all_data) - n_train]
        )
        test_data = scRNADataset(
            data_dir=test_data_dir,
            gene_order_path=gene_order_path,
        )
    elif mode == "RNA-protein":
        all_data = spMultiDataset(
            data_dir_1=train_data_dir,
            data_dir_2=train_data_dir_1,
            mask_dir=mask_dir
        )
        n_train = int(len(all_data) * 0.9)
        train_data, valid_data = torch.utils.data.random_split(
            all_data, [n_train, len(all_data) - n_train]
        )
        test_data = scMultiDataset(
            data_dir_1=test_data_dir,
            data_dir_2=test_data_dir_1,
        )
    return train_data, valid_data, test_data


def main(config):
    print(config)
    pl.seed_everything(config["seed"])

    save_dir = config["save_path"]
    os.makedirs(save_dir, exist_ok=True)

    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    gene_order_path = "/public/home/hpc214701031/project/scTrans/data/RNA/gtf/gene_order.csv"
    train_data, valid_data, test_data = load_data(
        mode=config["mode"], 
        train_data_dir=config["train_data_path"],
        test_data_dir=config["test_data_path"],
        train_data_dir_1=config["train_data_path_1"],
        test_data_dir_1=config["test_data_path_1"],
        gene_order_path=gene_order_path,
        mask_dir=config["mask_data_path"] if "mask_data_path" in config else None
    )

    nw = 8
    train_sampler = None
    valid_sampler = None
    test_sampler = None
    train_dataloader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=train_sampler is None,
        num_workers=nw,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=valid_sampler,
        # collate_fn=collator,
        drop_last=False,
        num_workers=nw,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=test_sampler,
        # collate_fn=collator,
        drop_last=False,
        num_workers=nw,
        pin_memory=True,
    )

    # Create model
    enc_config_model = HyenaConfig(
        d_model=config["enc_dim"],  
        emb_dim=config["position_emb_dim"], 
        max_seq_len=config["enc_vocab_len"], 
        vocab_len=config["enc_vocab_len"],
        n_layer=config["enc_depth"],
        output_hidden_states=False,
    )
    dec_config_model = HyenaConfig(
        d_model=config["dec_dim"],
        emb_dim=config["position_emb_dim"],
        max_seq_len=config["dec_vocab_len"],
        vocab_len=config["dec_vocab_len"],
        n_layer=config["dec_depth"],
        output_hidden_states=False,
    )
    if config["mode"] == "RNA-protein":
        if config["encoder_checkpoint_path"] is not None and config["decoder_checkpoint_path"] is not None:
            model = scTrans(
                enc_config_model,
                dec_config_model,
                post=True,
                encoder_ckpt_path=config["encoder_checkpoint_path"],
                decoder_ckpt_path=config["decoder_checkpoint_path"],
                lr=config["lr"],
            )
        else:
            print("No pre-trained model found. Error!")
    else:
        model = scTrans(enc_config_model, dec_config_model, 
                        mask_prob=config["mask_prob"], 
                        lr=config["lr"], emb_dropout=0.0)

    wandb_logger = TensorBoardLogger(
        save_dir="./10M/",
        name=args["wandb_name"]
    )

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss_epoch',
        filename='10M-{epoch:02d}-{valid_loss_epoch:.6f}',
        save_top_k=5,
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true", accelerator="gpu", devices=config["device"],
        logger=wandb_logger,
        max_epochs=config["epochs"],
        val_check_interval=1.0,
        default_root_dir=weights_dir,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    torch.distributed.destroy_process_group()
    if trainer.global_rank == 0:
        trainer = pl.Trainer(accelerator='gpu', devices=[0], logger=wandb_logger)
        trainer.test(model, test_dataloader)


if __name__ == "__main__":
    args = parser_args()
    main(args)
