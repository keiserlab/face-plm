import torch
from transformers import T5Tokenizer, T5EncoderModel
from typing import List, Dict
from tqdm import tqdm

def embed_sequences_prott5(sequences: List[str],
                           model_name: str = 'Rostlab/prot_t5_xl_bfd',
                           all_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
    """
    Generate per-amino-acid embeddings for a list of protein sequences using the ProtT5 model.

    Parameters
    ----------
    sequences : List[str]
        A list of protein sequences composed of uppercase amino acid letters.
    model_name : str, optional
        The Hugging Face model name or path to the ProtT5 model. Default is 'Rostlab/prot_t5_xl_bfd'.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays containing the embeddings for each amino acid.
    """
    # Load the ProtT5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()

    # Check if a GPU is available and move the model to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Dictionary to store embeddings
    embeddings = []
    # Process each sequence individually
    print(f"Embedding sequences with {model_name}...")
    for sequence in tqdm(sequences):
        # Replace rare/ambiguous amino acids by X and introduce white-space between all amino acids
        sequence = ' '.join(list(sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X')))
        # Tokenize the input sequence
        tokenized_input = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
        # Move tokenized input to the same device as the model
        input_ids = tokenized_input['input_ids'].to(device)
        attention_mask = tokenized_input['attention_mask'].to(device)
        # Generate embeddings
        with torch.no_grad():
            if all_hidden_states:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                embed = torch.cat(outputs.hidden_states).squeeze().cpu().numpy()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Extract per-amino-acid embeddings from the encoder's last hidden state
                embed = outputs.last_hidden_state.squeeze().cpu().numpy()
        # Store the embeddings in the dictionary
        embeddings.append(embed)

    return embeddings
