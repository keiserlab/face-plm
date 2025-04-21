uv venv .embed --python=3.10
uv pip install . --python=.embed
uv pip install esm --python=.embed
uv pip install "torchvision=0.19.0" --python=.embed