# get absolute path of this file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=$SCRIPT_DIR/../data

source  $SCRIPT_DIR/../.embed/bin/activate

uv run --python .embed --active \
    $SCRIPT_DIR/../src/face_plm/embed_gen/embed_all_layers.py \
    --zarr_dir $DATA_DIR/ankh_full_layers.zarr \
    --model_names ankh-large ankh-base \
    --path_to_csv $DATA_DIR/data/adk_evo-scale_dataset.csv