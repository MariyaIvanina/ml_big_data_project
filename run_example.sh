#!/usr/bin/env bash
./aggregate.py --data_dir history --begin_date 01.06.2017 --end_date 01.11.2017 --output_filename agg
echo "Data is aggregated! Compiling ..."
g++ -I Eigen/ -std=c++11 -O3 trmf.cpp -o trmf
echo "Compiled! Starting TRMF ..."
./trmf --input_file agg_values.csv --output_file result.txt