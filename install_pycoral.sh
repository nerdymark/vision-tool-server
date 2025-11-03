#!/bin/bash
#  Install PyCoral for Google Coral USB Accelerator

set -e

cd /home/mark/vision-tool-server
source venv/bin/activate

echo "Installing PyCoral dependencies..."
pip install Cython

echo "Cloning PyCoral repository..."
cd /tmp
rm -rf pycoral
git clone https://github.com/google-coral/pycoral.git
cd pycoral

echo "Installing PyCoral..."
bash scripts/install_requirements.sh python3
make wheel
pip install dist/pycoral-*-*.whl

echo "PyCoral installed successfully!"
