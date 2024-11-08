#!/bin/bash

# Remove the old CSV file to start fresh
rm -f tsp_dp_execution_times.csv

# Define schedule types, number of nodes, and thread counts
schedules=("static" "dynamic" "guided")
nodes=(4 6 8 10 12 14 16 18 20)
threads=(2 4 6 8)

# Generate and save a graph for each node count
for n in "${nodes[@]}"; do
    echo "Generating graph for $n nodes..."
    ./tsp_dp $n 1 "static"  # Save graph for each node count
done

# Run the program for each combination
for n in "${nodes[@]}"; do
    for t in "${threads[@]}"; do
        for sched in "${schedules[@]}"; do
            echo "Running with $n nodes, $t threads, $sched schedule..."
            ./tsp_dp $n $t $sched
        done
    done
done
