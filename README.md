# Parallelization of Travelling Salesman Problem
## Sequential
## Parallel
### OpenMP
  - Maximum number of nodes = 20 <br />
  - Commands to run: <br />
      &ensp;gcc -fopenmp tsp_dp_parallel.c -o tsp_dp <br />
      &ensp;bash run_experiments.sh <br />
      &ensp;python3 plot_results.py <br />
  - Static schedule seems optimal for larger number of nodes
    
