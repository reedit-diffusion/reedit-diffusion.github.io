#!/bin/bash

conda activate llava
python preprocess-llava.py --directory data
cd LLaVA
python edit.py --img_fol ../data --res_fol ../llava_results
python get_caption.py --img_fol ../data --res_fol ../llava_results
python3 truncate_caption.py --res_fol ../llava_results
cd ..
conda deactivate

echo "Completed generating llava caption!"

conda activate reedit
python3 preprocess.py --data_path data
python3 pnp.py --name reedit --group reedit

echo "Completed running ReEdit!"

conda deactivate