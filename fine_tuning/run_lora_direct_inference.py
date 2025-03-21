import hydra
from face_plm.inference.utils import load_model_from_wandb_for_inference
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd
import wandb
import regex as re
from transformers import AutoTokenizer
from argparse import ArgumentParser


def run_regression_inference_lora(artifact_id: dict,
                            config_name: str, 
                            split_num: int,
                            model_type: str = "latest",
                            config_path: str = "training/config/",
                            gpu_num: int = 0,
                            agg: str = None):

    with hydra.initialize(version_base=None, config_path=str(config_path)):
        config = hydra.compose(config_name=config_name)

    model_config = config.model_config
    data_config = config.data_config
    
    tokenizer = AutoTokenizer.from_pretrained(data_config.model_name)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'}) # Add the cls token to the tokenizer
    num_tokens = len(tokenizer)
    model_config.encoder.num_tokens = num_tokens
    model = load_model_from_wandb_for_inference(config=config,
                                            wandb_entity= "kampmann_lab",
                                            wandb_uid=artifact_id,
                                            model_type="best")

    data_module = hydra.utils.instantiate(data_config, split_num=split_num, tokenizer=tokenizer)

    train_preds = []
    train_labels = []
    val_preds = []
    val_labels = []

    model.eval()
    model.to(device=1)
    for batch in data_module.train_dataloader():
        *features, targets = batch
        features = [f.to(device=1) for f in features]
        
        with torch.no_grad():
            output = model(features).cpu().numpy()
        train_preds.append(output)
        train_labels.append(targets.numpy())

    for batch in data_module.val_dataloader():
        *features, targets = batch
        features = [f.to(device=1) for f in features]

        with torch.no_grad():
            output = model(features).cpu().numpy()
        val_preds.append(output)
        val_labels.append(targets.numpy())

    train_preds = np.concatenate(train_preds)
    train_labels = np.concatenate(train_labels)
    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)

    train_spearman = spearmanr(train_preds, train_labels)[0].item()
    val_spearman = spearmanr(val_preds, val_labels)[0].item()
    train_rmse = root_mean_squared_error(train_preds, train_labels)
    val_rmse = root_mean_squared_error(val_preds, val_labels)
    train_r2 = r2_score(train_labels, train_preds)
    val_r2 = r2_score(val_labels, val_preds)
    train_pearson = pearsonr(np.squeeze(train_preds), train_labels)[0].item()
    val_pearson = pearsonr(np.squeeze(val_preds), val_labels)[0].item()
    return train_spearman, val_spearman, train_rmse, val_rmse, train_r2, val_r2, train_pearson, val_pearson


def main():
    # setting up argparser
    parser = ArgumentParser()
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    api = wandb.Api()
    filter_expression = re.compile(".*kcat_f.*")
    runs = api.runs(args.wandb_project)
    artifact_rows = []
    for run in runs: 
        if (filter_expression.search(run.name) 
            and run.state == "finished" 
        ):
            artifact_dict = {"name": run.name,
                "id": run.id, 
                "split_num": run.config["data"]["fold_num"],
                "model_name": run.config["data"]["model_name"],
                "hydra_config": run.config["hydra_config"],
                }
            if "dropout" in run.name:
                artifact_dict["lora_dropout"] = run.config["encoder"]["lora_dropout"]
            else: 
                artifact_dict["lora_dropout"] = 0.1
            artifact_rows.append(artifact_dict)
            
    artifact_df = pd.DataFrame(artifact_rows)

    rows = []
    num_models = len(artifact_df)
    for i, row in list(artifact_df.iterrows()):
        print(f"Running inference on {i+1}/{num_models}")
        model_name = row["model_name"]
        config_name = row["hydra_config"]
        artifact_id = row["id"]
        split_num = row["split_num"]
        
        train_spearman, val_spearman, train_rmse, val_rmse, train_r2, val_r2, train_pearson, val_pearson= run_regression_inference_lora(artifact_id=artifact_id,
                                config_path="config/",
                                config_name=config_name,
                                split_num=split_num,
                                model_type="best",
                                gpu_num=1
                                )

        rows.append({"model": model_name,
                        "fold_num": split_num,
                        "split": "train",
                        "spearman": train_spearman,
                        "rmse": train_rmse,
                        "r2": train_r2,
                        "pearson": train_pearson,
                        "dropout": row["lora_dropout"]})
        rows.append({"model": model_name,
                        "fold_num": split_num, 
                        "split": "val",
                        "spearman": val_spearman,
                        "rmse": val_rmse,
                        "r2": val_r2,
                        "pearson": val_pearson,
                        "dropout": row["lora_dropout"]})
    results_df = pd.DataFrame(rows)
    if args.output_dir.enswith("/"):
        results_df.to_csv(f"{args.output_dir}lora_direct_ft_kcat_dropout.csv", index=False)
    else:
        results_df.to_csv(f"{args.output_dir}/lora_direct_ft_kcat_dropout.csv", index=False)


if __name__ == "__main__":
    main()
