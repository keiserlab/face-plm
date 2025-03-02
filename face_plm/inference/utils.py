import wandb
import torch
import hydra

def load_latest_model_artifact(project_name,
                               entity,
                               wandb_uid,
                               model_type,
                               artifact_type="model",
                               save_dir="./"
                               ):
    """
    Load the latest version of a model artifact from WandB.
    
    Parameters
    ----------
    project_name : str
        The name of the project where the model artifact is stored.
    entity : str
        The entity where the project is stored.
    wandb_uid : str
        The unique identifier of the model artifact.
    artifact_type : str, optional
        The type of the artifact to load, by default "model".
    save_dir: str, optional
        The directory where the model will be saved, by default "/"
    
    Returns
    -------
    dict
        The state dictionary of the model.
    """
    # Initialize WandB API
    api = wandb.Api()
    model_name = f"model-{wandb_uid}"
    # Construct the artifact path (e.g., 'entity/project/model_name:latest')
    artifact_path = f"{entity}/{project_name}/{model_name}:{model_type}" if entity else f"{project_name}/{model_name}:{model_type}"
    print(f"Fetching latest artifact from: {artifact_path}")
    # Load the latest artifact
    artifact = api.artifact(artifact_path, type=artifact_type)
    # Download and get the directory where the model is saved
    if save_dir.endswith("/"):
        save_dir = save_dir[:-1]
    artifact_dir = artifact.download(root=f"{save_dir}/wandb_artifacts/{model_name}_{model_type}")
    # Assuming your model file is named "model.ckpt" in the artifact
    model_file_path = f"{artifact_dir}/model.ckpt"
    checkpoint = torch.load(model_file_path, map_location="cpu")
    # Load the model using torch.load (you may need to adapt this if you have a custom loading function)
    state_dict = checkpoint["state_dict"]
    
    print(f"Model loaded from {model_file_path}")
    return state_dict


def load_model_from_wandb_for_inference(config,
                                        wandb_entity,
                                        wandb_uid,
                                        model_type="latest",
                                        save_dir="./"):
    """
    This function loads a model from WandB for inference.

    Parameters
    ----------
    config : OmegaConf Dict
        The configuration for the model.
    wandb_entity : str
        The entity where the model is stored.
    wandb_uid : str
        The unique identifier of the model.
    
    Returns
    -------
    torch.nn.Module
        The loaded model.
    """
    model = hydra.utils.instantiate(config.model_config)
    state_dict = load_latest_model_artifact(config.wandb_config.project,
                                            wandb_entity,
                                            wandb_uid, 
                                            model_type,
                                            save_dir=save_dir)
    model.load_state_dict(state_dict)
    model.eval()
    return model