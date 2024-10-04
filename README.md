## Parallel
  ### OpenMP
    - Maximum number of nodes = 20
    - Commands to run:
        gcc -fopenmp tsp_dp_parallel.c -o tsp_dp
        bash run_experiments.sh
        python3 plot_results.py
    - Static schedule seems optimal for larger number of nodes
    
