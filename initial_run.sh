#!/bin/bash

# Change directory to LLaVA
cd LLaVA

# Create new conda environment
# Check if llava environment exists
if conda env list | grep -q "llava"; then
    echo "LLaVA environment already exists"
else
    conda create -n llava python=3.10 -y
fi

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llava

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -e .
pip install protobuf
cd ..
python preprocess-llava.py --directory data
cd LLaVA
python edit.py --img_fol ../data --res_fol ../llava_results
python get_caption.py --img_fol ../data --res_fol ../llava_results
python3 truncate_caption.py --res_fol ../llava_results
cd ..
conda deactivate

echo "Completed generating llava caption!"

# Check if pnp environment exists
if conda env list | grep -q "reedit"; then
    echo "reedit environment already exists"
else
    conda create -n reedit python=3.9 -y
fi

conda activate reedit
pip install -r requirements.txt

python3 preprocess.py --data_path data
python3 pnp.py --name reedit --group reedit