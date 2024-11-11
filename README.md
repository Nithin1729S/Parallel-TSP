# Parallelization of Travelling Salesman Problem
## Sequential Execution
![tsp_combined_execution_times](https://github.com/user-attachments/assets/2cc12c1c-b7de-47a0-9bce-8c3035b944ed)


## Parallel
### OpenMP
![Execution_Time_vs_Number_of_Nodes](https://github.com/user-attachments/assets/575cb5f1-920f-49fc-912b-a1abdd8cfca0)
![Execution_Time_vs_Number_of_Threads](https://github.com/user-attachments/assets/a7e0d6d0-5eb0-4f35-9f60-90a8a101bfb4)


  - Maximum number of nodes = 20 <br />
  - Commands to run: <br />
      &ensp;gcc -fopenmp tsp_dp_parallel.c -o tsp_dp <br />
      &ensp;bash run_experiments.sh <br />
      &ensp;python3 plot_results.py <br />
  - Static schedule seems optimal for larger number of nodes

### MPI

![tsp_dp_mpi](https://github.com/user-attachments/assets/b168d912-19aa-4f01-9a2d-2b24aba5750a)

### CUDA

![execution_time_comparison](https://github.com/user-attachments/assets/02c2b6f0-e09b-4f93-9076-f13338a380b4)
