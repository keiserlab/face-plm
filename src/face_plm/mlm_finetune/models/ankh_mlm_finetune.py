
from torch.optim import AdamW
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM
from face_plm.probes.utils import HyperParameterScheduler


def create_lora_model(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    bias: str = "none",
):
    """
    Load the Ankh large model and wrap it in a LoRA PEFT model.

    Parameters
    ----------
    lora_r : int, optional
        The LoRA rank parameter.
    lora_alpha : int, optional
        The alpha scaling factor for LoRA.
    lora_dropout : float, optional
        The dropout probability for LoRA.
    bias : str, optional
        Whether to add bias parameters during LoRA adaptation.

    Returns
    -------
    model : nn.Module
        The PEFT LoRA-wrapped Ankh model.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained("ElnaggarLab/ankh-base")
    config = LoraConfig(
        r=lora_r, 
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
    )
    peft_model = get_peft_model(model, config)
    return peft_model


class AnkhLoRAFineTune(pl.LightningModule):
    """
    PyTorch Lightning module for finetuning the LoRA-wrapped Ankh model.
    
    Parameters
    ----------
    model : nn.Module
        The PEFT LoRA-wrapped Ankh model.
    tokenizer : ankh.tokenizer
        The tokenizer associated with the model.
    learning_rate : float, optional
        Learning rate for AdamW optimization.
    """
    def __init__(self,
                 lr_scheduler: HyperParameterScheduler,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        super().__init__()
        self.model = create_lora_model(lora_r, lora_alpha, lora_dropout)
        self.lr_scheduler = lr_scheduler
        # Save hyperparameters to enable checkpointing of hyperparameters
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels=None):
        """
        Forward pass of the model.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Tokenized input IDs.
        attention_mask : torch.FloatTensor
            Attention masks indicating which tokens are valid.
        decoder_input_ids : torch.LongTensor
            Tokenized decoder input IDs.
        labels : torch.LongTensor, optional
            Labels for computing the language modeling loss.

        Returns
        -------
        outputs : transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
            The output of the forward pass containing the loss if labels are provided.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Parameters
        ----------
        batch : dict
            A batch of data containing 'input_ids', 'attention_mask', and 'labels'.
        batch_idx : int
            The index of the batch within the training epoch.

        Returns
        -------
        loss : torch.Tensor
            The computed loss value for the batch.
        """
        self._update_optimizer()
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        outputs = self.forward(input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        lr = self.lr_scheduler.schedule[self.global_step]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Parameters
        ----------
        batch : dict
            A batch of data containing 'input_ids', 'attention_mask', and 'labels'.
        batch_idx : int
            The index of the batch within the validation epoch.

        Returns
        -------
        loss : torch.Tensor
            The computed validation loss.
        """
        input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        outputs = self.forward(input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            The optimizer (AdamW) used for training.
        """
        optimizer = AdamW(self.parameters())
        return optimizer
    
    def _update_optimizer(self) -> None:
        optimizer = self.optimizers(use_pl_optimizer=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_scheduler.schedule[self.global_step]
    
    def on_train_start(self) -> None:
        """
        Train start hook for scheduler computation.
        """
        # Computing the hyperparameter schedule
        assert self.trainer.max_epochs is not None, "The maximum number of epochs must be specified."
        train_steps = self.trainer.num_training_batches * self.trainer.max_epochs
        self.lr_scheduler.compute_schedule(self.trainer.train_dataloader)
        schedule_length = len(self.lr_scheduler.schedule)
        assert schedule_length != 0 and train_steps <= schedule_length
