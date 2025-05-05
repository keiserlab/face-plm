# get absolute path of this file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG_NAME=$1

echo "Output directory: $OUTPUT_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Current directory: $(pwd)"
python $SCRIPT_DIR/../fine_tuning/launch_finetune_mlm.py \
    --config $CONFIG_NAME