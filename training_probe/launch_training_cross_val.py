import os
from pathlib import Path

import pytorch_lightning as pl
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from pytorch_lightning.loggers import WandbLogger
import hydra
import tempfile
import wandb

from face_plm.probes.utils import clean_hydra_config_for_wandb



def main():
    # Seeding everything to ensure reproducibility
    pl.seed_everything(1)
    
    # Argparsing
    desc = "Script for Training a regressor model on protein sequence"
    parser = ArgumentParser(
        description=desc, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="config file name without .yaml extension")
    parser.add_argument("--log_dir", type=Path, help="Directory to save logs")
    args = parser.parse_args()
    config_path = Path(__file__).parent / "config"
    
    full_path = Path.cwd() / config_path
    relative_path = os.path.relpath(full_path, Path(__file__).parent)

    with hydra.initialize(version_base=None, config_path=str(relative_path)):
        config = hydra.compose(config_name=str(args.config))

    log_dir = tempfile.mkdtemp(dir=args.log_dir)

    #train_config = config.train_config
    model_config = config.model_config
    data_config = config.data_config

    for i in range(5):
        # Instantiating the train config with defaults
        train_config = hydra.utils.instantiate(config.train_config)

        # Generating DataLoaders
        print("Generating Training and Validation DataSets...")
        data_module = hydra.utils.instantiate(data_config, split_num=i)
        
        # Instantiating Model
        print("Instantiating the model...")
        model = hydra.utils.instantiate(model_config)

        # Initializing Wandb Logger
        wandb_config = config.wandb_config
        wandb_config = hydra.utils.instantiate(wandb_config)
        
        logger = WandbLogger(project=wandb_config.project,
                            name=wandb_config.run_name + f"_fold_{i}",
                            log_model=True,
                            group=wandb_config.group_name,
                            entity=wandb_config.entity,
                            tags=wandb_config.tags,
                            dir=log_dir,
                            save_dir=log_dir)

        # Creating Trainer from argparse args
        if train_config.gpu_num == -1:
            gpu_num = -1
        else:
            gpu_num = [train_config.gpu_num]

        
        callbacks = [pl.callbacks.ModelCheckpoint(
                        save_last=True,
                        save_top_k=1,
                        monitor="val_loss",
                        mode="min",)
                    ]

        assert torch.cuda.is_available(), "CUDA is not available and is required for training"

        trainer = pl.Trainer(accelerator="gpu",
                            devices=gpu_num,
                            logger=logger,
                            enable_checkpointing=True,
                            profiler="simple",
                            #callbacks=callbacks,
                            max_epochs=train_config.max_epoch_number,
                            log_every_n_steps=train_config.log_steps,
                            precision=32)
        # Training the model
        print("Training Model...")
        trainer.fit(model, datamodule=data_module)

        cleaned_config = clean_hydra_config_for_wandb(config)
        cleaned_config["hydra_config"] = str(args.config)
        cleaned_config["data"]["fold_num"] = i
        wandb.config.update(cleaned_config, allow_val_change=True)
        wandb.finish()

    
if __name__ == "__main__":
    main()