#!/bin/bash


lang="$1"  # Set language here


# Define the parameters
device="cuda"
num_workers=4
yaml_conf="conf/resnet293.yaml"
output_path="./data/$lang"


# Update val_name in the YAML file with the chosen language
sed -i "s|^val_name:.*|val_name: './data/$lang'|" "$yaml_conf"

mode='extract'  # Modes: 'extract', 'evaluate'
python ./eval_RF5.py --num_workers=${num_workers} --device=${device} --yaml_path=${yaml_conf} --output_path=${output_path} --mode=${mode}

mode='evaluate'  # Modes: 'extract', 'evaluate'
python ./eval_RF5.py --num_workers=${num_workers} --device=${device} --yaml_path=${yaml_conf} --output_path=${output_path} --mode=${mode}

