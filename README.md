# MHP2_eval

This repo contains the evaluation scripts for the multi-human parsing task in the MHP-V2 dataset (https://lv-mhp.github.io/)

## Contents:
mhp_data.py: This script generates a list of data, and also visualizes the dataset

eval_mhp.py: This script evaluates the predictios. It generates a set of perfect predictions with the ground truth, and evluates the perfect predictions. To evaluate your algorithm, replace the results['MASKS'] and results['DETS'] with the output of your algorithm.

eval_sumission.py: This script takes in the format of submission (https://lv-mhp.github.io/evaluate) and outputs the metrics. 

voc_eval.py: A helper script. 
