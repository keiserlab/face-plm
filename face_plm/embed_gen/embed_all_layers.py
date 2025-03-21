import zarr
import numpy as np
import pandas as pd
import argparse
from face_plm.embed_gen.ankh_utils import embed_sequences_ankh


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--zarr_dir', type=str, required=True)
    # Adding argument where you can list the model names to include
    arg_parser.add_argument('--model_names', type=str, nargs='+', required=True)
    arg_parser.add_argument('--path_to_csv', type=str, required=True)
    args = arg_parser.parse_args()
    model_names = args.model_names
    zarr_dir = args.zarr_dir
    path_to_csv = args.path_to_csv
    valid_model_names = ["ankh-large",
                         "ankh-base"]
    # finding model names that are not valid
    invalid_model_names = [x for x in model_names if x not in valid_model_names]
    assert all([x in valid_model_names for x in model_names]), f"Invalid model names: {invalid_model_names}"

    # reading in the sequences from the csv
    data_df = pd.read_csv(path_to_csv)
    sequences = data_df['sequence'].values

    # create root for zarr file
    zarr_root = zarr.open(zarr_dir, mode='w')
    for model_name in np.unique(model_names):
        # create zarr group for each model
        model_group = zarr_root.create_group(name=model_name, overwrite=True)
        assert model_name in ["ankh-large", "ankh-base"], f"Model {model_name} not available. Choose from {valid_model_names}"
        seq_embeds = embed_sequences_ankh(sequences, model_name, all_hidden_states=True)
        for seq, seq_embed in zip(sequences, seq_embeds):
            # create zarr array for each sequence
            model_group.create_dataset(name=seq, data=seq_embed, dtype=np.float32)


if __name__ == '__main__':
    main()
