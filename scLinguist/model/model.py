from math import ceil
import torch
import scanpy as sc
from torch import nn as nn
import pytorch_lightning as pl
from scLinguist.model.modeling_hyena import HeynaModel
from scLinguist.model.tokenizer import mask_data_ADT, mask_data_RNA


class scHeyna_dec(nn.Module):
    def __init__(self, config, emb_dropout=0.0, tie_embed=False):
        super().__init__()
        self.to_vector = nn.Linear(1, config.d_model)
        # self.to_vector_tech = nn.Linear(1, config.d_model)
        self.transformer = HeynaModel(config)
        self.to_out = nn.Linear(config.d_model, 1) if not tie_embed else None
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, inputs, return_encodings=False):
        x = inputs[:, :, :]
        b, n = x.shape[0], x.shape[1]
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=2)
            x = self.to_vector(x)
        x = self.dropout(x)
        x = self.transformer(inputs_embeds=x)
        if return_encodings:
            return x
        return torch.squeeze(self.to_out(x))

class scHeyna_enc(nn.Module):
    def __init__(self, config, emb_dropout=0.0, tie_embed=False):
        super().__init__()
        self.to_vector = nn.Linear(1, config.d_model)
        # self.to_vector_tech = nn.Linear(1, config.d_model)
        self.transformer = HeynaModel(config)
        self.to_out = nn.Linear(config.d_model, 1) if not tie_embed else None
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, inputs, return_encodings=False):
        x = inputs
        b, n = x.shape[0], x.shape[1]
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=2)
            x = self.to_vector(x)

        x = self.dropout(x)
        x = self.transformer(inputs_embeds=x)
        if return_encodings:
            return x
        return torch.squeeze(self.to_out(x))



class MLPTranslator(nn.Module):
    """
    from scTranslator https://github.com/TencentAILabHealthcare/scTranslator
    Class description: translator from RNA to protein
    fully connected layer with adjustable number of layers and variable dropout for each layer

    """

    # ----- Define all layers -----#
    def __init__(
            self,
            num_fc_input,
            num_output_nodes,
            num_fc_layers,
            initial_dropout,
            act=nn.ReLU(),
            **kwargs
    ):
        super(MLPTranslator, self).__init__(**kwargs)
        fc_d = pow(
            num_fc_input / num_output_nodes, 1 / num_fc_layers
        )  # reduce factor of fc layer dimension
        # --- Fully connected layers ---#
        self.num_fc_layers = num_fc_layers
        if num_fc_layers == 1:
            self.fc0 = nn.Linear(num_fc_input, num_output_nodes)
        else:
            # the first fc layer
            self.fc0 = nn.Linear(num_fc_input, int(ceil(num_fc_input / fc_d)))
            self.dropout0 = nn.Dropout(initial_dropout)
            if num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                self.fc1 = nn.Linear(int(ceil(num_fc_input / fc_d)), num_output_nodes)
            else:
                # the middle fc layer
                for i in range(1, num_fc_layers - 1):
                    tmp_input = int(ceil(num_fc_input / fc_d ** i))
                    tmp_output = int(ceil(num_fc_input / fc_d ** (i + 1)))
                    exec("self.fc{} = nn.Linear(tmp_input, tmp_output)".format(i))
                    if (
                            i < ceil(num_fc_layers / 2)
                            and 1.1 ** (i + 1) * initial_dropout < 1
                    ):
                        exec(
                            "self.dropout{} = nn.Dropout(1.1**(i+1)*initial_dropout)".format(
                                i
                            )
                        )
                    elif (
                            i >= ceil(num_fc_layers / 2)
                            and 1.1 ** (num_fc_layers - 1 - i) * initial_dropout < 1
                    ):
                        exec(
                            "self.dropout{} = nn.Dropout(1.1**(num_fc_layers-1-i)*initial_dropout)".format(
                                i
                            )
                        )
                    else:
                        exec("self.dropout{} = nn.Dropout(initial_dropout)".format(i))
                # the last fc layer
                exec(
                    "self.fc{} = nn.Linear(tmp_output, num_output_nodes)".format(i + 1)
                )

        # --- Activation function ---#
        self.act = act

    # ----- Forward -----#
    def forward(self, x):
        # x size:  [batch size, feature_dim]

        if self.num_fc_layers == 1:
            outputs = self.fc0(x)
        else:
            # the first fc layer
            outputs = self.act(self.dropout0(self.fc0(x)))
            if self.num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                outputs = self.fc1(outputs)
            else:
                # the middle fc layer
                for i in range(1, self.num_fc_layers - 1):
                    outputs = eval(
                        "self.act(self.dropout{}(self.fc{}(outputs)))".format(i, i)
                    )
                # the last fc layer
                outputs = eval("self.fc{}(outputs)".format(i + 1))

        return outputs


class scTrans(pl.LightningModule):
    def __init__(self, 
            enc_ret_config, 
            dec_ret_config, 
            mode="RNA",
            encoder_ckpt_path= None,
            decoder_ckpt_path = None,
            mask_prob=0.3, 
            lr=1e-4, 
            emb_dropout=0.0
        ):
        """
        :param enc_ret_config: config for encoder
        :param dec_ret_config: config for decoder
        :param mode: "RNA", "protein", "RNA-protein
        :param encoder_ckpt_path: path to the pre-trained encoder checkpoint
        :param decoder_ckpt_path: path to the pre-trained decoder checkpoint
        :param mask_prob: probability of masking the input data
        :param lr: learning rate
        :param emb_dropout: dropout rate for the embedding layer
        """
        if mode not in ["RNA", "protein", "RNA-protein"]:
            raise ValueError("mode must be one of 'RNA', 'protein', or 'RNA-protein'")
        super().__init__()
        self.mode = mode
        self.mask_prob = mask_prob
        self.lr = lr
        self.encoder = scHeyna_enc(
            enc_ret_config, emb_dropout=emb_dropout, tie_embed=True
        )
        self.decoder = scHeyna_dec(
            dec_ret_config, emb_dropout=emb_dropout
        )
        if (encoder_ckpt_path != None) and (decoder_ckpt_path != None):
            self.load_encoder_decoder(encoder_ckpt_path, decoder_ckpt_path)
            self.translator = MLPTranslator(enc_ret_config.max_seq_len, dec_ret_config.max_seq_len, 2, 0.1)
        else:
            self.translator = None
        self.epoch_train_loss_list = []
        self.epoch_val_loss_list = []
        self.test_emb_list = []
        self.test_pred = []
        self.save_hyperparameters()

    def load_encoder_decoder(self, encoder_ckpt_path, decoder_ckpt_path):
        encoder_checkpoint = torch.load(encoder_ckpt_path)["state_dict"]
        encoder_state_dict = {
            k[len("encoder."):]: v
            for k, v in encoder_checkpoint.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state_dict, strict=False)

        decoder_checkpoint = torch.load(decoder_ckpt_path)["state_dict"]
        decoder_state_dict = {
            k[len("decoder."):]: v
            for k, v in decoder_checkpoint.items()
            if k.startswith("decoder.")
        }
        self.decoder.load_state_dict(decoder_state_dict, strict=False)


    def forward(self, seq_in):
        encodings = self.encoder(
            seq_in, return_encodings=True
        )  # batch_size, input_seq_lenth, dim
        if self.translator == None:
            return encodings, self.decoder(encodings)
        else:
            seq_out = (
                self.translator(encodings.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
            return encodings, seq_out, self.decoder(seq_out)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def recon_loss(self, data, pred_data, mask_idx=None):
        if mask_idx is None:
            return nn.functional.mse_loss(pred_data, data, reduction='mean')
        else:
            loss1 = self.masked_recon_loss(pred_data, data, mask_idx)
            return loss1

    def masked_recon_loss(self, data, pred_data, mask_idx=None):
        masked_squared_error = ((data - pred_data) ** 2) * mask_idx
        summed_error_per_row = masked_squared_error.sum(dim=1)
        mask_count_per_row = mask_idx.sum(dim=1).float()
        mask_count_per_row[mask_count_per_row == 0] = 1
        average_error_per_row = summed_error_per_row / mask_count_per_row
        recon_loss = average_error_per_row.mean()
        return recon_loss


    def training_step(self, batch, batch_idx):
        if self.mode=="RNA-protein":
            x, y, mask_final = batch[0], batch[1], batch[2]
            embed, embed_mlp, recon = self(x)
        elif self.mode=="RNA":
            y = batch
            mask_x, mask_final = mask_data_RNA(batch, self.mask_prob)
            embed, recon = self(mask_x)
        elif self.mode == "protein":
            y, b = batch
            mask_x, mask_idx, mask_final = mask_data_ADT(batch, self.mask_prob)
            embed, recon = self(mask_x)
        recon_loss = self.recon_loss(y, recon, mask_final)

        metrics = {
            "train/loss": recon_loss,
        }
        for key, value in metrics.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )


        return recon_loss


    def validation_step(self, batch, batch_idx):
        if self.mode=="RNA-protein":
            x, y, mask_final = batch[0], batch[1], batch[2]
            embed, embed_mlp, recon = self(x)
        elif self.mode=="RNA":
            y = batch
            mask_x, mask_final = mask_data_RNA(batch, self.mask_prob)
            embed, recon = self(mask_x)
        elif self.mode == "protein":
            y, b = batch
            mask_x, mask_idx, mask_final = mask_data_ADT(batch, self.mask_prob)
            embed, recon = self(mask_x)
        recon_loss = self.recon_loss(y, recon, mask_final)

        metrics = {
            "valid_loss": recon_loss,
        }

        for key, value in metrics.items():
            self.log(
                key,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return {"val_loss": recon_loss}


    def test_step(self, batch, batch_idx):
        if self.mode=="RNA-protein":
            x, y, mask_final = batch[0], batch[1], batch[2]
            embed, embed_mlp, recon = self(x)
        elif self.mode=="RNA":
            y = batch
            mask_x, mask_final = mask_data_RNA(batch, self.mask_prob)
            embed, recon = self(mask_x)
        elif self.mode == "protein":
            y, b = batch
            mask_x, mask_idx, mask_final = mask_data_ADT(batch, self.mask_prob)
            embed, recon = self(mask_x)
        recon_loss = self.recon_loss(y, recon, mask_final)

        metrics = {
            "test/loss": recon_loss,
        }

        for key, value in metrics.items():
            self.log(
                key,
                value,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        self.test_emb_list.append(self.embed.detach().to("cpu").numpy())

        return {"test_loss": recon_loss}

    def cluster(self, embed, reso=0.1):
        adata = sc.AnnData(embed)
        sc.pp.neighbors(adata, n_neighbors=20)
        sc.tl.louvain(adata, resolution=reso, random_state=0)
        return adata.obs["louvain"].astype(int).to_numpy()

    def plot_gene_umap(self, emb, reso=0.5, name=""):
        adata = sc.AnnData(emb)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
        sc.tl.umap(adata)
        sc.tl.louvain(adata, resolution=reso, random_state=0)
        adata.obs["label"] = adata.obs["louvain"].astype(int)
        adata.obs["label"] = adata.obs["label"].astype("category")
        fig = sc.pl.umap(adata, color=["label"], return_fig=True, show=False)
        self.logger.experiment.add_figure("umap_pros"+name, fig)

