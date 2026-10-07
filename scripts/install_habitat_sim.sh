#!/usr/bin/env bash
# Build and install Habitat-sim on Ubuntu with sudo access.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
HABITAT_SIM_DIR=${HABITAT_SIM_DIR:-../habitat-sim}
HABITAT_SIM_SHA=2bc5b4573b37202cb44b771ded34d5992a93f795

sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build libjpeg-dev libglm-dev \
  libegl1-mesa-dev libgl1-mesa-dev xorg-dev freeglut3-dev

if [[ ! -d .habitat-build ]]; then
  uv venv .habitat-build --python 3.12
fi
uv pip install --python .habitat-build/bin/python \
  'scikit-build-core>=0.10' 'pybind11>=2.10' 'numpy>=2.0' setuptools wheel pip
if [[ ! -d "$HABITAT_SIM_DIR" ]]; then
  git clone https://github.com/facebookresearch/habitat-sim.git "$HABITAT_SIM_DIR"
fi
git -C "$HABITAT_SIM_DIR" checkout "$HABITAT_SIM_SHA"
[[ $(git -C "$HABITAT_SIM_DIR" rev-parse HEAD) == "$HABITAT_SIM_SHA" ]]
# Audio is disabled, so skip its submodule.
git -C "$HABITAT_SIM_DIR" config submodule.src/deps/rlr-audio-propagation.update none
git -C "$HABITAT_SIM_DIR" submodule update --init --recursive
# Allow the locked NumPy and Pillow versions in the wheel metadata.
sed -i 's/numpy==1\.26\.4/numpy>=2.0/g' "$HABITAT_SIM_DIR/pyproject.toml"
sed -i 's/pillow==10\.4\.0/pillow>=10.4.0/g' "$HABITAT_SIM_DIR/pyproject.toml"
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) HABITAT_BUILD_GUI_VIEWERS=OFF \
  HABITAT_WITH_CUDA=OFF HABITAT_WITH_AUDIO=OFF HABITAT_WITH_BULLET=OFF \
  .habitat-build/bin/python -m pip wheel "$HABITAT_SIM_DIR" --no-deps --no-build-isolation \
  --wheel-dir="$HABITAT_SIM_DIR/dist"
uv pip install --python .venv/bin/python "$HABITAT_SIM_DIR"/dist/habitat_sim-*.whl
