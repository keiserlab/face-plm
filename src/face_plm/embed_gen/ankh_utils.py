from typing import List
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def embed_sequences_ankh(sequences: List[str],
                         model_name: str = 'ankh-large',
                         all_hidden_states: bool = False) -> torch.Tensor:
    """
    Generate per-amino-acid embeddings for a given protein sequence using the Ankh model.

    Parameters
    ----------
    sequence : List[str]
        The protein sequence composed of uppercase amino acid letters.
    model_name : str, optional
        The model variant to use: 'ankh-large' or 'ankh-base'. Default is 'ankh-large'.

    Returns
    -------
    List[torch.Tensor]
        A tensor of shape (sequence_length, embedding_dim) containing the embeddings for each amino acid.
    """
    model_name = model_name.lower()
    valid_model_names = ['ankh-large', 'ankh-base']
    # Load the specified Ankh model and tokenizer
    if model_name == 'ankh-large':
        # model, tokenizer = ankh.load_large_model()
        model = AutoModelForSeq2SeqLM.from_pretrained("ElnaggarLab/ankh-large")
        tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-large")
    elif model_name == 'ankh-base':
        # model, tokenizer = ankh.load_base_model()
        model = AutoModelForSeq2SeqLM.from_pretrained("ElnaggarLab/ankh-base")
        tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
    else:
        raise ValueError(f"Invalid model name: {model_name}. Choose from {valid_model_names}")
    embeddings = []
    # Tokenize the input sequence
    print(f"Embedding sequences with {model_name}...")
    for seq in tqdm(sequences):
        outputs = tokenizer(seq, return_tensors='pt')
        # Generate embeddings
        with torch.no_grad():
            out = model(input_ids=outputs['input_ids'],
                        decoder_input_ids=outputs['input_ids'],
                        attention_mask=outputs['attention_mask'],
                        output_hidden_states=True)
        # Extract per-amino-acid embeddings from the encoder's last hidden state
        if all_hidden_states:
            embed = torch.cat(out.encoder_hidden_states).squeeze().cpu().numpy()
        else:
            embed = out.encoder_last_hidden_state.squeeze(0).cpu().numpy()
        embeddings.append(embed)

    return embeddings