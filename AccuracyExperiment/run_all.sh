#!/usr/bin/env bash

./run_exp1a.sh &
./run_exp1b.sh &
./run_exp1c.sh &

wait || true

shutdown now
