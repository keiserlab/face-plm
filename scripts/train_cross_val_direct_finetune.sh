# get absolute path of this file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG_NAME=$1

python $SCRIPT_DIR/../fine_tuning/launch_finetune_regression.py \
    --config $CONFIG_NAME 