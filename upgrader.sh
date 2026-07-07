git pull --ff-only;

source source .venv/bin/activate;

python -m pip install --upgrade pip;
pip install -e .;

brew update;
brew upgrade ffmpeg chromaprint;

pytest - q;