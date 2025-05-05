# get absolute path of this file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR=$1

python $SCRIPT_DIR/../training_probe/launch_notorch_probes.py \
    --output_dir $OUTPUT_DIR