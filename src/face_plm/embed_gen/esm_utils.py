from typing import List
import torch
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig

def embed_sequences_esm(sequences: List[str],
                        model_name: str,
                        device: str = 'gpu',
                        token = None) -> List[torch.Tensor]:
    available_esm_models = []
    if model_name not in ["esmc_600m", "esm3-sm-open-v1"]:
        raise ValueError(f"Model {model_name} not available. Choose from {available_esm_models}")
    # Load the ESM model
    proteins = [ESMProtein(sequence=seq) for seq in sequences]
    if 'esmc' in model_name:
        model = ESMC.from_pretrained("esmc_600m").to("cuda")
    elif 'esm3' in model_name:
        model = ESM3.from_pretrained("esm3-sm-open-v1").to("cuda")
    proteins = [ESMProtein(sequence=seq) for seq in sequences]
    embeddings = []
    print(f"Embedding sequences with {model_name}...")
    if "esmc" in model_name:
        with torch.no_grad():
            encoded_seqs = [model.encode(p).sequence for p in proteins]
            for seq in tqdm(encoded_seqs):
                embed = model(seq.unsqueeze(0)).embeddings.to(torch.float32)
                embeddings.append(embed.cpu().squeeze().numpy())
    elif "esm3" in model_name:
        for protein in proteins:
            with torch.no_grad():
                protein_tensor = model.encode(protein)
                # Configure sampling to return per-residue embeddings
                sampling_config = SamplingConfig(return_per_residue_embeddings=True)
                # Perform forward pass and sampling
                output = model.forward_and_sample(protein_tensor, sampling_config)
                embed = output.per_residue_embedding
                embeddings.append(embed.cpu().squeeze().numpy())
    del model
    return embeddings