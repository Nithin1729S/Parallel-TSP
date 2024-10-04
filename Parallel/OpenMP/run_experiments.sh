#!/bin/bash

# Remove the old CSV file to start fresh
rm -f tsp_dp_execution_times.csv

# Define schedule types, number of nodes, and thread counts
schedules=("static" "dynamic" "guided")
nodes=(4 6 8 10 12 14 16 18 20)
threads=(2 4 6 8)

# Run the program for each combination
for n in "${nodes[@]}"; do
    for t in "${threads[@]}"; do
        for sched in "${schedules[@]}"; do
            echo "Running with $n nodes, $t threads, $sched schedule..."
            ./tsp_dp $n $t $sched
        done
    done
done
