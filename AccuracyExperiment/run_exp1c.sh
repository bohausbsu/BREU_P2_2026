#!/usr/bin/env bash

set -euo pipefail

mkdir -p ae_acc

for i in {1..10}; do
	echo "Run $i of 10"
	python Experiment1c.py --seed "$i" --data-root mnist --flag-frac 0.1 --at-r 0.01 --out-csv "ae_acc_$i.csv" --out "ae_acc_$i.png"
	mv ae_acc_* ae_acc/
done

python combine_results.py --results-dir ae_acc --out-csv ae_acc/ae_acc_avg.csv --out-png ae_acc_avg.png --title "SPECTRA AE Detection Scores Averaged Over 10 Runs" --pattern "ae_acc_*.csv"
