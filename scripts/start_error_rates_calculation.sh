#!/bin/sh
cd ..
python main.py \
    --find_error_rates \
    --distances_file_path resultsLFW/rgb/distances.csv \
    --results_dir ./resultsLFW/rgb