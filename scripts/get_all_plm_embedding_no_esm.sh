# get absolute path of this file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/../data


python $SCRIPT_DIR/../src/face_plm/embed_gen/embed.py \
    --zarr_dir $DATA_DIR/adk_plm_embeddings.zarr \
    --model_names ankh-large ankh-base prot_t5_xl_bfd ProstT5 \
    --path_to_csv $DATA_DIR/adk_evo-scale_dataset.csv 

