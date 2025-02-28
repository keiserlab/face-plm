import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from toolkit import RegressorModel,  RegressionDataset
from sklearn.model_selection import StratifiedKFold

# spearmanr
from scipy.stats import spearmanr

### parameters
RANDOM_SEED = 42
optimizer= "Adam"
batch_size=32
lr=1e-6
wd=0.01
dropout_rate=0.05
hidden_layers=1
hidden_dim=256
max_epochs=3000
patience=20
###

dataset_path = "~/code/adk-deep-learning/data/esm_lidtype_activity_dataset.csv"

df = pd.read_csv(dataset_path)

# drop organism column
df = df.drop(['org_name'], axis=1)

np.random.seed(RANDOM_SEED)
# five fold cross val split of dataset
kf = StratifiedKFold(n_splits=5)
loss_dict = {}


class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_loss.append(float(trainer.callback_metrics["val_loss"]))

loss_dict = {}
spearman_dict = {}

for i, (train_index, val_index) in enumerate(kf.split(df, df["lid_type"])):
    print("Training Fold: ", i)
    
    train = df.iloc[train_index]
    val = df.iloc[val_index]

    train_dataset = RegressionDataset(train.iloc[:, :-1].to_numpy())
    val_dataset = RegressionDataset(val.iloc[:, :-1].to_numpy())
    pl.seed_everything(RANDOM_SEED)

    model = RegressorModel(train_dataset, 
                val_dataset,
                optimizer=optimizer,
                batch_size=batch_size,
                lr=lr,
                wd=wd,
                dropout_rate=dropout_rate,
                hidden_layers=hidden_layers,
                hidden_dim=hidden_dim,)
            
    loss_log = LossLogger()

    # Defining Callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=patience,
                                        verbose=False,
                                        mode='min')

    # Creating Trainer from argparse args
    trainer = pl.Trainer(accelerator="gpu",
                        devices=[0], 
                        callbacks=[early_stop_callback, loss_log],
                        #profiler="simple",
                        max_epochs=max_epochs,
                        log_every_n_steps=1,
                        )

    # # Training the model
    trainer.fit(model) 
    model.eval()
    y_pred = model(train_dataset.X)
    y_pred = y_pred.detach().numpy()
    y_true = train_dataset.y

    y_val_pred = model(val_dataset.X)
    y_val_pred = y_val_pred.detach().numpy()
    y_val_true = val_dataset.y
    # print(f"Train Spearman: {spearmanr(y_true, y_pred)}")
    # print(f"Validation Spearman: {spearmanr(y_val_true, y_val_pred)}")
    spearman_dict[i] = {"train_spearman": float(spearmanr(y_true, y_pred)[0]), "val_spearman": float(spearmanr(y_val_true, y_val_pred)[0])}

    #loss_dict[i] = {"train_loss": loss_log.train_loss, "val_loss": loss_log.val_loss}
for key, value in spearman_dict.items():
    print(key, value)