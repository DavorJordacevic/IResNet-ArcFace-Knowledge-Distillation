#!/bin/sh
cd ..
python main.py \
    --find_EER \
    --error_rates_file_path resultsLFW/rgb/error_rates.json \
    --results_dir ./resultsLFW/rgb