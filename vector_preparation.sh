#!/usr/bin/env bash
./forecast.py --data_dir history --begin_date 01.06.2017 --end_date 01.08.2017 --output_file predicted.csv --output_f_file F2.csv
mv tmp_agg_names.csv tmp_agg_2_names.csv
./forecast.py --data_dir history --begin_date 01.06.2017 --end_date 01.12.2017 --output_file predicted.csv --output_f_file F6.csv
mv tmp_agg_names.csv tmp_agg_6_names.csv
./forecast.py --data_dir history --begin_date 01.06.2017 --end_date 01.02.2018 --output_file predicted.csv --output_f_file F9.csv
mv tmp_agg_names.csv tmp_agg_9_names.csv
./forecast.py --data_dir history --begin_date 01.06.2017 --end_date 01.12.2017 --output_file predicted.csv --columns O,C,H,L,V,BV --output_f_file FBV.csv
mv tmp_agg_names.csv tmp_aggBV_names.csv