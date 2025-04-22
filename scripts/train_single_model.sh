# get absolute path of this file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG_NAME=$1

python $SCRIPT_DIR/../training_probe/launch_training.py \
    --config $CONFIG_NAME 