from typing import Union
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np 
from adk_dl.utils import TrainConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
import zarr
from transformers import AutoTokenizer

# TODO: need to make this relative in some manner for other users
# Constants
SEQUENCE_DATA_DF_PATH = "/srv/home/dmuir/code/adk-deep-learning/data/adk_evo-scale_dataset.csv"
EMBEDDING_ZARR_PATH = "/srv/ds/set-1/group/duncan_parker/embedding_zarr_files/adk_plm_embeddings.zarr"
SPLIT_INFO_JSON = "/srv/home/dmuir/code/adk-deep-learning/data/adk_5fold_cross_val_orgs.json"
ANKH_FULL_LAYER_ZARR_PATH = "/srv/ds/set-1/group/duncan_parker/embedding_zarr_files/ankh_full_layers.zarr"

def generate_adk_seq_datasets(dataset_path, random_seed=42, train_org_names=None, val_org_names=None):
    df = pd.read_csv(dataset_path)

    if (train_org_names is not None) and (val_org_names is not None):
        # print(len(train_org_names), len(val_org_names))
        # print(len(df["org_name"].unique()))
        #assert len(train_org_names) + len(val_org_names) == len(df["org_name"].unique()), "Organism names do not match"
        train = df.set_index("org_name").loc[train_org_names].reset_index()
        val = df.set_index("org_name").loc[val_org_names].reset_index()
        train = train.drop(['org_name'], axis=1)
        val = val.drop(['org_name'], axis=1)
    else:
        # drop organism column
        df = df.drop(['org_name'], axis=1)
        train, val = train_test_split(df, test_size=0.2, random_state=random_seed, stratify=df["lid_type"])

    train_dataset = RegressionDataset(train.iloc[:, :-1].to_numpy())
    val_dataset = RegressionDataset(val.iloc[:, :-1].to_numpy())

    return train_dataset, val_dataset


def generate_dataloader(train_config: TrainConfig,
                        dataset: Dataset,
                        shuffle: bool = True) -> DataLoader:
    """
    This function generates a PyTorch DataLoader object for training a model.

    Parameters
    ----------
    train_config : TrainConfig
        An object containing the training configuration parameters.
    dataset : Dataset
        The dataset object to be used for training.
    
    Returns
    -------
    DataLoader
        A PyTorch DataLoader object for training the model.
    """
    num_workers = train_config.num_workers
    batch_size = train_config.batch_size
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataloader


class PLMEmbeddingRegressionDataset(Dataset):
    def __init__(self,
                 model_name: str,
                 train_test: str = "train",
                 split_num: int = 1,
                 regression_col_name: str = "log10_kcat",
                 aggregation: Union[None, str]=None):
        """
        This class is a PyTorch Dataset that reads in protein embeddings for protein sequences.

        Parameters
        ----------
        model_name : str
            The name of the model used to generate the embeddings.
        train_test : str, optional
            The type of data to load: 'train' or 'test'. Default is 'train'.
        split_num : int, optional
            The split number to load. Default is 1.
        aggregation : Union[None, str], optional
            The type of aggregation to apply to the embeddings: 'mean', 'max', 'min', or None. Default is None.
            If None, then padding is performed and full embeddings are returned.
        """
        assert aggregation in [None, "mean", "max", "min"]
        assert train_test in ["train", "test"]
        with open(SPLIT_INFO_JSON) as f:
            split_info = json.load(f)
        split = str(split_num)
        assert split in split_info.keys()
        self.data_df = pd.read_csv(SEQUENCE_DATA_DF_PATH)
        org_names = split_info[split][train_test]
        self.data_df = self.data_df[self.data_df["org_name"].isin(org_names)]
        # Reading in the zarr root
        self.zarr_root = zarr.open(EMBEDDING_ZARR_PATH, mode="r")
        self.model_name = model_name
        self.sequences = self.data_df["sequence"].values
        self.aggregation = aggregation
        self.y = self.data_df[regression_col_name].values
        if aggregation is None:
            max_seq_length, embedding_dim = self._get_dataset_info()
            self.max_seq_length = max_seq_length
            self.padding_token = torch.nn.Parameter(torch.zeros(1, embedding_dim))

    def _get_dataset_info(self):
        max_seq_length = 0
        embedding_dim = self.zarr_root[self.model_name][self.sequences[0]].shape[1]
        for seq in self.zarr_root[self.model_name].keys():
            seq_embed = self.zarr_root[self.model_name][seq]
            seq_length = seq_embed.shape[0]
            if seq_length > max_seq_length:
                max_seq_length = seq_length
        return max_seq_length, embedding_dim

    def _pad_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        seq_length = seq.shape[0]
        padding_length = self.max_seq_length - seq_length
        padding = torch.cat([seq, self.padding_token.repeat(padding_length, 1)], dim=0)
        return padding

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        aa_embeddings = self.zarr_root[self.model_name][seq]
        aa_embeddings = torch.Tensor(np.array(aa_embeddings))   # shape: [seq_length, embedding_dim]
        y = torch.Tensor(np.array(self.y[idx]).astype('float32'))
        if self.aggregation is None:
            X = self._pad_sequence(aa_embeddings)  # shape: [max_seq_length, embedding_dim]
            return X, y
        elif self.aggregation == "mean":
            X = aa_embeddings.mean(dim=0)  # shape: [embedding_dim]
            return X, y
        elif self.aggregation == "max":
            X = aa_embeddings.max(dim=0).values  # shape: [embedding_dim]
            return X, y
        elif self.aggregation == "min":
            X = aa_embeddings.min(dim=0).values  # shape: [embedding_dim]
            return X, y


class PLMEmbeddingRegressionDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_config: TrainConfig,
                 model_name: str,
                 split_num: int = 1,
                 aggregation: Union[None, str]=None):
        super().__init__()
        self.train_config = train_config
        self.model_name = model_name
        self.aggregation = aggregation
        self.split_num = split_num
        self.train_ds = PLMEmbeddingRegressionDataset(self.model_name, train_test="train", split_num=split_num, aggregation=self.aggregation)
        self.val_ds = PLMEmbeddingRegressionDataset(self.model_name, train_test="test", split_num=split_num, aggregation=self.aggregation)

    def train_dataloader(self):
        return generate_dataloader(self.train_config, self.train_ds)

    def val_dataloader(self):
        return generate_dataloader(self.train_config, self.val_ds, shuffle=False)


class PLMSeq2RegressionDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 train_test="train",
                 split_num=0,
                 model_name="ElnaggarLab/ankh-base",
                 target="log10_kcat"):
        """
        Pytorch Dataset for Seq2Regression tasks using pretrained language models

        Parameters
        ----------
        train_test : str
            "train" or "test"
        split_num : int
            The split number to use
        model_name : str
            The name of the model to use
        target : str
            The target column to use
        """
        
        train_test = train_test.lower()

        # load cross val split dict
        with open(SPLIT_INFO_JSON) as f:
            split_info = json.load(f)
        split = str(split_num)
        assert split in split_info.keys()

        # load sequence and target data
        self.data_df = pd.read_csv(SEQUENCE_DATA_DF_PATH)
        # compute max length of sequences
        max_length = max(self.data_df['sequence'].apply(len))
        # filter data based on split
        org_names = split_info[split][train_test]
        self.data_df = self.data_df[self.data_df["org_name"].isin(org_names)]

        # get sequences and target values
        self.sequences = self.data_df['sequence'].values
        self.target = self.data_df[target].values

        # load tokenizer
        self.tokenizer = tokenizer

        self.max_length = max_length + 1 # +1 for the [CLS] token
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # prepend [CLS] token to sequence
        sequence = "[CLS]"+ self.sequences[idx]  
        inputs = self.tokenizer(sequence,
                                max_length=self.max_length,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True)
        labels = torch.tensor(self.target[idx], dtype=torch.float)
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), labels.squeeze()
       

class PLMSeq2RegressionDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_config: TrainConfig,
                 model_name: str,
                 tokenizer: AutoTokenizer,
                 split_num: int = 1):
        super().__init__()
        self.train_config = train_config
        self.model_name = model_name
        self.split_num = split_num
        self.train_ds = PLMSeq2RegressionDataset(train_test="train", split_num=split_num, model_name=model_name, tokenizer=tokenizer)
        self.val_ds = PLMSeq2RegressionDataset(train_test="test", split_num=split_num, model_name=model_name, tokenizer=tokenizer)

    def train_dataloader(self):
        return generate_dataloader(self.train_config, self.train_ds)

    def val_dataloader(self):
        return generate_dataloader(self.train_config, self.val_ds, shuffle=False)


class AnkhLayerSpecificRegressionDataset(Dataset):
    def __init__(self,
                 layer: int =-1,
                 aggregation: Union[None, str] = None,
                 y_col: str = "log10_kcat",
                 train_test: str = "train",
                 split_num: int = 0):

        split = str(split_num)
        with open(SPLIT_INFO_JSON) as f:
            split_info = json.load(f)

        
        assert split in split_info.keys()
        assert train_test in ["train", "test"]
        assert aggregation in ["mean", "max", "min", None], f"Aggregation {aggregation} not in ['mean', 'max', 'min', None]"


        self.aggregation = aggregation
        self.layer = layer
        self.split_info = split_info
        self.zarr_path = ANKH_FULL_LAYER_ZARR_PATH


        csv_file = SEQUENCE_DATA_DF_PATH
        self.data_df = pd.read_csv(csv_file)
        org_names = split_info[split][train_test]
        self.data_df = self.data_df[self.data_df["org_name"].isin(org_names)]

        
        self.seqs = self.data_df["sequence"].values
        self.zarr_root = zarr.open(self.zarr_path, mode="r")
       
        self.encoder = self.zarr_root["ankh-base"]
        
        
        possible_layers = list(range(self.encoder[self.seqs[0]].shape[0]))
        possible_layers += [-1]
        assert self.layer in possible_layers, f"Layer {self.layer} not in {possible_layers}"
        self.y = self.data_df[y_col].values
        if aggregation is None:
            max_seq_length, embedding_dim = self._get_dataset_info()
            self.max_seq_length = max_seq_length
            self.padding_token = torch.nn.Parameter(torch.zeros(1, embedding_dim))

    def _get_dataset_info(self):
        max_seq_length = 0
        embedding_dim = self.encoder[self.seqs[0]][self.layer].shape[1]
        for seq in self.encoder.keys():
            seq_embed = self.encoder[seq][self.layer]
            seq_length = seq_embed.shape[0]
            if seq_length > max_seq_length:
                max_seq_length = seq_length
        return max_seq_length, embedding_dim

    def _pad_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        seq_length = seq.shape[0]
        padding_length = self.max_seq_length - seq_length
        padding = torch.cat([seq, self.padding_token.repeat(padding_length, 1)], dim=0)
        return padding
        
    def __len__(self):
        return len(self.seqs)

    @staticmethod
    def apply_layernorm_to_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        embedding_dim = embeddings.size(-1)
        layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim, elementwise_affine=False)
        # Apply the layer normalization to the embeddings
        normalized_embeddings = layer_norm(embeddings)
        return normalized_embeddings

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        y = self.y[idx]
        aa_embeddings = self.encoder[seq][self.layer]
        aa_embeddings = torch.Tensor(np.array(aa_embeddings))   # shape: [seq_length, embedding_dim]
        y = torch.Tensor(np.array(self.y[idx]).astype('float32'))
        if self.aggregation is None:
            X = self._pad_sequence(aa_embeddings)  # shape: [max_seq_length, embedding_dim]
            X = self.apply_layernorm_to_embeddings(X)
            return X, y
        elif self.aggregation == "mean":
            X = aa_embeddings.mean(dim=0)  # shape: [embedding_dim]
            return X, y
        elif self.aggregation == "max":
            X = aa_embeddings.max(dim=0).values  # shape: [embedding_dim]
            return X, y
        elif self.aggregation == "min":
            X = aa_embeddings.min(dim=0).values  # shape: [embedding_dim]
            return X, y


class AnkhLayerSpecificRegressionDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_config: TrainConfig,
                 layer: int =-1,
                 aggregation: Union[None, str] = None,
                 y_col: str = "log10_kcat",
                 split_num: int = 0):
        super().__init__()
        self.train_config = train_config
        self.layer = layer
        self.aggregation = aggregation
        self.y_col = y_col
        self.split_num = split_num
        self.train_ds = AnkhLayerSpecificRegressionDataset(train_test="train",
                                                            split_num=split_num,
                                                            aggregation=aggregation,
                                                            layer=layer)
        self.val_ds = AnkhLayerSpecificRegressionDataset(train_test="test",
                                                            split_num=split_num,
                                                            aggregation=aggregation,
                                                            layer=layer)

    def train_dataloader(self):
        return generate_dataloader(self.train_config, self.train_ds)

    def val_dataloader(self):
        return generate_dataloader(self.train_config, self.val_ds, shuffle=False)