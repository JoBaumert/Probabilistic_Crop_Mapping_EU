#!/bin/bash

#whenever an error occurs, stop script immediately and don't run other jobs
set -e

echo run model_parameter_estimation.py
python generation_of_prob_crop_map/model_parameter_estimation.py 

echo run calculation_of_prior_crop_probabilities.py
python generation_of_prob_crop_map/calculation_of_prior_crop_probabilities.py 
echo run incorporation_of_aggregated_info.py
python generation_of_prob_crop_map/incorporation_of_aggregated_info.py 

echo run parquet_to_raster.py
python data_preparation_and_analysis/parquet_to_raster.py 

echo run visualize_cropmap.py
python data_preparation_and_analysis/visualize_cropmap.py 
