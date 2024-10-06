#!/bin/bash

# Remove the old CSV file to start fresh
rm -f tsp_dp_execution_times.csv

# Define schedule types, number of nodes, and thread counts
schedules=("static" "dynamic" "guided" "runtime")
nodes=(4 6 8 10 12 14 16 18 20)
threads=(2 4 6 8)

# OMP_SCHEDULE values to test for runtime scheduling
omp_schedules=("static" "dynamic" "guided")

# Run the program for each combination
for n in "${nodes[@]}"; do
    for t in "${threads[@]}"; do
        for sched in "${schedules[@]}"; do
            if [ "$sched" == "runtime" ]; then
                for omp_sched in "${omp_schedules[@]}"; do
                    echo "Running with $n nodes, $t threads, runtime schedule (OMP_SCHEDULE=$omp_sched)..."
                    export OMP_SCHEDULE="$omp_sched"
                    ./tsp_dp $n $t $sched
                done
            else
                echo "Running with $n nodes, $t threads, $sched schedule..."
                ./tsp_dp $n $t $sched
            fi
        done
    done
done
