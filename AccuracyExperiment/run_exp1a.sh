#!/usr/bin/env bash

set -euo pipefail

mkdir -p ffnn_acc

for i in {1..10}; do
	echo "Run $i of 10"
	python Experiment1a.py --seed "$i" --dataset college_student_placement_dataset.csv --target-col Placement --flag-frac 0.1 --at-r 0.01 --out-csv "ffnn_acc_$i.csv" --out "ffnn_acc_$i.png"
	mv ffnn_acc_* ffnn_acc/
done

python combine_results.py --results-dir ffnn_acc --out-csv ffnn_acc/ffnn_acc_avg.csv --out-png ffnn_acc_avg.png --title "SPECTRA FFNN Detection Scores Averaged Over 10 Runs" --pattern "ffnn_acc_*.csv"
