SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

uv run $SCRIPT_DIR/../training_probe/launch_training.py --config esmc_600m-agg_mlp