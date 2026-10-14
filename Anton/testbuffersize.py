import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from time import time



comm = MPI.COMM_WORLD
# Get rank in the `comm` communicator.

rank = comm.Get_rank()
# Get total number of MPI ranks in the `comm` communicator.
cores = comm.Get_size()
status = MPI.Status()

m = 1000
problem_size = np.linspace(1,257)
results = []

for problem in problem_size:

    total_time = 0

    if rank == 0:
        print(problem)
        message = np.zeros(int(np.ceil(problem*1e3/8)),dtype = np.float64)
        tiny_mess = np.empty(1, dtype = np.int16)
        t1 = time()
        comm.Send(message, dest = 1, tag = 1)
        comm.Recv(tiny_mess, source = 1, status = status, tag = 0)
        t2 = time()

        total_time += t2-t1
    else:
        buffer = np.empty(int(np.ceil(problem*1e3/8)),dtype = np.float64)
        comm.Send(np.array([1]), dest= 0, tag = 0)
        comm.Recv(buffer, source = 0, status = status, tag = 1)

    if rank == 0:
        results.append(total_time)

if rank == 0:
    plt.plot(problem_size, results, marker="o")
    plt.xlabel("Problem size")
    plt.ylabel("Total send time (s)")
    plt.title("MPI send time versus problem size")
    plt.tight_layout()
    plt.savefig("results_problem_plot.png")
    plt.close()
    
