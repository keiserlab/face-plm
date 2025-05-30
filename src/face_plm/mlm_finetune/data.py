from torch.utils.data import Dataset
from transformers import AutoTokenizer
from face_plm.probes.data import generate_dataloader
from face_plm.probes.utils import TrainConfig
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path


# Getting path of this file
file_path = Path(__file__).resolve()
# Getting the parent directory of this file
parent_dir = file_path.parent
# defining the data directory
data_dir = str(parent_dir) + "/../../data/"
EC27_DATA = f"{data_dir}uniprotkb_ec_2_7_2025_01_05_org_filtered.tsv"
ADK_DATA = f"{data_dir}uniprot_adk.txt"
ADK_DATA_SUBSET = f"{data_dir}uniprot_adk_subset.csv"


class AnkhEC27MLMDataset(Dataset):
    def __init__(self,
                 max_length=512,
                 mask_ratio=0.1,
                 train_test="train"):
        train_test = train_test.lower()
        tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
        ec27_df = pd.read_csv(EC27_DATA, sep='\t')
        # 20% validation split
        np.random.seed(42)
        train_ids = np.random.choice(np.arange(0, len(ec27_df)), int(0.8*len(ec27_df)), replace=False)
        val_inds = np.array([i for i in np.arange(0, len(ec27_df)) if i not in train_ids])
        if train_test == "train":
            ec27_df = ec27_df.iloc[train_ids]
        else:
            ec27_df = ec27_df.iloc[val_inds]
        ec27_df['seq_len'] = ec27_df['Sequence'].apply(lambda x: len(x))
        ec27_df = ec27_df.loc[ec27_df['seq_len'] <= max_length]
        self.sequences = ec27_df['Sequence'].values
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence,
                                max_length=self.max_length,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True)
        masked_inds = np.random.choice(np.arange(self.max_length), int(self.mask_ratio * self.max_length), replace=False)

        # converting the mask indices to tge <extra_id_#> token
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        masked_input_ids = inputs['input_ids'].clone()
        mask[masked_inds] = 1
        mask = mask.bool().unsqueeze(0)
        for i, mask_ind in enumerate(masked_inds):
            masked_input_ids[0, mask_ind] = self.tokenizer.convert_tokens_to_ids(f'<extra_id_{i}>')
        labels = inputs['input_ids'].detach().clone()
        labels[mask] = inputs['input_ids'][mask]
        return masked_input_ids.squeeze(), inputs['attention_mask'].squeeze(), labels.squeeze()


class AnkhFineTuneDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_config: TrainConfig,
                 max_length: int = 512,
                 mask_ratio: float = 0.1):
        super().__init__()
        self.train_config = train_config
        self.max_length = max_length
        self.mask_ratio = mask_ratio

    def train_dataloader(self):
        train_ds = AnkhEC27MLMDataset(max_length=self.max_length, mask_ratio=self.mask_ratio, train_test="train")
        return generate_dataloader(self.train_config, train_ds, shuffle=True)
    

class AnkhADKMLMDataset(Dataset):
    def __init__(self,
                 max_length=512,
                 mask_ratio=0.1,
                 train_test="train"):
        train_test = train_test.lower()
        tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
        data_df = pd.read_csv(ADK_DATA, sep='\t')
        # 20% validation split
        np.random.seed(42)
        train_ids = np.random.choice(np.arange(0, len(data_df)), int(0.8*len(data_df)), replace=False)
        val_inds = np.array([i for i in np.arange(0, len(data_df)) if i not in train_ids])
        if train_test == "train":
            data_df = data_df.iloc[train_ids]
        else:
            data_df = data_df.iloc[val_inds]
        data_df['seq_len'] = data_df['Sequence'].apply(lambda x: len(x))
        data_df = data_df.loc[data_df['seq_len'] <= max_length]
        self.sequences = data_df['Sequence'].values
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence,
                                max_length=self.max_length,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True)
        masked_inds = np.random.choice(np.arange(self.max_length), int(self.mask_ratio * self.max_length), replace=False)
        # converting the mask indices to tge <extra_id_#> token
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        masked_input_ids = inputs['input_ids'].clone()
        mask[masked_inds] = 1
        mask = mask.bool().unsqueeze(0)
        for i, mask_ind in enumerate(masked_inds):
            masked_input_ids[0, mask_ind] = self.tokenizer.convert_tokens_to_ids(f'<extra_id_{i}>')
        labels = inputs['input_ids'].detach().clone()
        labels[mask] = inputs['input_ids'][mask]
        return masked_input_ids.squeeze(), inputs['attention_mask'].squeeze(), labels.squeeze()


class AnkhADKFineTuneDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_config: TrainConfig,
                 max_length: int = 512,
                 mask_ratio: float = 0.1):
        super().__init__()
        self.train_config = train_config
        self.max_length = max_length
        self.mask_ratio = mask_ratio

    def train_dataloader(self):
        train_ds = AnkhADKMLMDataset(max_length=self.max_length, mask_ratio=self.mask_ratio, train_test="train")
        return generate_dataloader(self.train_config, train_ds, shuffle=True)


class AnkhADKSubsetMLMDataset(Dataset):
    def __init__(self,
                 max_length=512,
                 mask_ratio=0.1):
        tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
        data_df = pd.read_csv(ADK_DATA_SUBSET)
        data_df['seq_len'] = data_df['Sequence'].apply(lambda x: len(x))
        data_df = data_df.loc[data_df['seq_len'] <= max_length]
        self.sequences = data_df['Sequence'].values
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence,
                                max_length=self.max_length,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True)
        masked_inds = np.random.choice(np.arange(self.max_length), int(self.mask_ratio * self.max_length), replace=False)
        # converting the mask indices to tge <extra_id_#> token
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        masked_input_ids = inputs['input_ids'].clone()
        mask[masked_inds] = 1
        mask = mask.bool().unsqueeze(0)
        for i, mask_ind in enumerate(masked_inds):
            masked_input_ids[0, mask_ind] = self.tokenizer.convert_tokens_to_ids(f'<extra_id_{i}>')
        labels = inputs['input_ids'].detach().clone()
        labels[mask] = inputs['input_ids'][mask]
        return masked_input_ids.squeeze(), inputs['attention_mask'].squeeze(), labels.squeeze()
    

class AnkhADKSubsetFineTuneDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_config: TrainConfig,
                 max_length: int = 512,
                 mask_ratio: float = 0.1):
        super().__init__()
        self.train_config = train_config
        self.max_length = max_length
        self.mask_ratio = mask_ratio

    def train_dataloader(self):
        train_ds = AnkhADKSubsetMLMDataset(max_length=self.max_length, mask_ratio=self.mask_ratio)
        return generate_dataloader(self.train_config, train_ds, shuffle=True)
