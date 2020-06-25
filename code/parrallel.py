#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:56:29 2020

@author: garethlomax
"""


from mpi4py import MPI
from multiprocessing import Process, cpu_count
 
def do_something_useful(rank, shared_process_number):
    # Do something useful here.
    print('Python hybrid, MPI_Process-local_process (not quite a thread): {}-{}'.format(rank, shared_process_number))
 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
 
print('Calling Python multiprocessing process from Python MPI rank {}'.format(rank))
 
# Create shared-ish processes
shared_processes = []
#for i in range(cpu_count()):
for i in range(8):
    p = Process(target=do_something_useful, args=(rank, i))
    shared_processes.append(p)
 
# Start processes
for sp in shared_processes:
    sp.start()
 
# Wait for all processes to finish
for sp in shared_processes:
    sp.join()
 
comm.Barrier()