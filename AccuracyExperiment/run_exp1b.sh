#!/usr/bin/env bash

set -euo pipefail

mkdir -p cnn_acc

for i in {1..10}; do
	echo "Run $i of 10"
	python Experiment1b.py --seed "$i" --data-root PetImages --flag-frac 0.1 --at-r 0.01 --out-csv "cnn_acc_$i.csv" --out "cnn_acc_$i.png"
	mv cnn_acc_* cnn_acc/
done

python combine_results.py --results-dir cnn_acc --out-csv cnn_acc/cnn_acc_avg.csv --out-png cnn_acc_avg.png --title "SPECTRA CNN Detection Scores Averaged Over 10 Runs" --pattern "cnn_acc_*.csv"
