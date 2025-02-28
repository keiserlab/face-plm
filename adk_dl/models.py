from torch.utils.data import  DataLoader
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.manifold import trustworthiness
from torchmetrics.regression import SpearmanCorrCoef
from adk_dl.utils import HyperParameterScheduler, create_lora_model
from abc import ABC, abstractmethod

####################
##### Encoders
####################

class FeedForwardEncoder(pl.LightningModule):
    def __init__(self, 
        input_dim: int, 
        hidden_dim: int, 
        hidden_layers: int, 
        hidden_taper: float = 1.0,
        dropout_rate: float = 0.0,
        activation_function = torch.nn.functional.relu):
        super().__init__()

        if hidden_layers < 1:
                raise Exception("The number of hidden layers must be at least 1")
        else:     
            model_layers = [torch.nn.Linear(input_dim, hidden_dim)] 
            for i in range(hidden_layers - 1):
                model_layers += [torch.nn.Linear(int(hidden_dim * (hidden_taper ** (i))), 
                int(hidden_dim * (hidden_taper ** (i+1))))] 
        self.layers = torch.nn.ModuleList(model_layers)
        self.activation_function = activation_function
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, dropout=False):
        h = x
        for layer in self.layers:
            h = self.activation_function(layer(h))
            if dropout:
                h = self.dropout(h)
        return h

class TransformerClsTokenEncoder(pl.LightningModule):
    def __init__(self, 
                 num_layers: int, 
                 num_heads: int, 
                 input_dim: int,
                 dim_feedforward: int = 2048,
                 dropout_rate: float = 0.0
                 ):
        super().__init__()
    
        self.input_dim = input_dim
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout_rate
        )
        self.transformer = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.input_dim))
        
    def forward(self, batch, dropout=False):
        # TODO: dropout currently goes nowhere
        # get batch embeddings
        h = batch
        # repeat cls_token and move to device
        cls_tokens = self.cls_token.repeat(h.shape[0], 1, 1).to(self.device)
        # concatenate cls_token and embeddings
        tokens = torch.column_stack((cls_tokens, h))
        # pass through transformer
        transformer_output = self.transformer(tokens)
        # get cls_token output
        cls_tokens = transformer_output[:,0,:]
        return cls_tokens

####################
##### Task Heads
####################

class RegressionHead(pl.LightningModule):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.head(x)

class ContinuousMetricHead(pl.LightningModule):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.head = torch.nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        return self.head(x)

####################
##### Generic Model
####################
class LightningModel(pl.LightningModule):
    """
    A generic PytorchLightning model.

    Paramters:
        lr_scheduler: HyperParameterScheduler
            The learning rate scheduler to use during training.
        encoder: torch.nn.ModuleList
            The encoder to use for the model.
        optimizer: str
            Name of the optimizer for use during training.
        wd: float
            Weight decay for use during training.
        dropout_rate: float
            Dropout rate for use during training.
    """
    def __init__(self,   
                 lr_scheduler: HyperParameterScheduler,
                 encoder: pl.LightningModule,
                 task_head: pl.LightningModule,
                 loss_func,
                 metric = None,
                 optimizer: str = 'Adam',
                 wd: float = 0.01,
                 dropout_rate: float = 0.1,
                 ):

        super(LightningModel, self).__init__()

        self.encoder = encoder
        self.task_head = task_head
        
        self.lr_scheduler = lr_scheduler
        self.wd = wd

        self.optimizer = optimizer
        # making sure requested optimizer is implemented
        self.configured_optim = ['RMSprop', 'Adam', 'SGD', 'Adagrad']
        assert self.optimizer in self.configured_optim, f"Optimizer must be one of: {self.configured_optim}"

        self.save_hyperparameters()
        self.loss_func = loss_func
        self.metric = metric

    def _update_optimizer(self) -> None:
        optimizer = self.optimizers(use_pl_optimizer=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_scheduler.schedule[self.global_step]

    def forward(self, batch, dropout=False):
        # TODO: dropout currently goes nowhere
        h = batch
        if isinstance(h, tuple) or isinstance(h, list):
            # unpack batch
            h = self.encoder(*h)
        else:
            h = self.encoder(h)
        return self.task_head(h)
        
    def predict(self, x):
        # predict outputs for input batch x
        self.eval()
        with torch.no_grad():
            y_hat = self(x)
        return y_hat

    def training_step(self, batch, _):
        # Update optimizer with learning rate schedule
        self._update_optimizer()
        # Get features and labels from batch
        *features, labels = batch
        # Forward pass
        pred = self.forward(features, dropout=True)
        # Calculating the loss
        loss = self.loss_func(pred, labels.unsqueeze(1).float())   
        # Logging loss and learning rate    
        self.log('train_loss', loss,on_step=False, on_epoch=True)
        self.log('learning_rate', self.lr_scheduler.schedule[self.global_step], on_step=True, on_epoch=False)

        # self.log('train_spearman', self.spearman(pred, labels.unsqueeze(1)), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        *features, labels = batch
        pred = self.forward(features)
        loss = self.loss_func(pred, labels.unsqueeze(1).float())
      
        self.log('val_loss', loss,on_step=False, on_epoch=True)
        # self.log('val_spearman', self.spearman(pred, labels.unsqueeze(1)), on_step=False, on_epoch=True)

    # Configuring optimizers
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=1e-3, weight_decay=self.wd)
        return optim_encoder

    def on_train_start(self) -> None:
        # Computing the hyperparameter schedule
        assert self.trainer.max_epochs is not None, "The maximum number of epochs must be specified."
        train_steps = self.trainer.num_training_batches * self.trainer.max_epochs
        self.lr_scheduler.compute_schedule(self.trainer.train_dataloader)
        schedule_length = len(self.lr_scheduler.schedule)
        assert schedule_length != 0 and train_steps <= schedule_length

class LoRAFinetunePLMEncoder(pl.LightningModule):
    """
    PyTorch Lightning module for finetuning a LoRA-wrapped PLM model.
    
    Parameters
    ----------
    model : nn.Module
        The PEFT LoRA-wrapped PLM model.
    tokenizer : tokenizer
        The tokenizer associated with the model.
   
    """
    def __init__(self,
                model_name: str, 
                num_tokens: int,
                lora_r: int = 16,
                lora_alpha: int = 32,
                lora_dropout: float = 0.1,
                ):
        super().__init__()
        self.lora_model = create_lora_model(model_name, lora_r, lora_alpha, lora_dropout)
        self.lora_model.resize_token_embeddings(num_tokens)

    def forward(self,
        input_ids,
        attention_mask, 
       ):
        """
        Forward pass of the model.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Tokenized input IDs.
        attention_mask : torch.FloatTensor
            Attention masks indicating which tokens are valid.
        labels : torch.LongTensor, optional
            Labels for computing the language modeling loss.

        Returns
        -------
        outputs : cls_tokens following the forward pass
        """
        embeddings = self.lora_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            output_hidden_states=True,
        )
        
        cls_tokens = embeddings.encoder_last_hidden_state[:, 0, :]
        
        return cls_tokens
