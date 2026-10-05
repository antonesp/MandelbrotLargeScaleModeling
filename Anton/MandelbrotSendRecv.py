import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from time import time

_help = f"""\
{sys.argv[0]} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]

Here are some examples:

Call it with a chunk-size of 10
$ {sys.argv[0]} 10

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ {sys.argv[0]} 10 100x500

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
spanning the coordinates x \\in 0.1-0.3 and y \\in 0.2-0.3
$ {sys.argv[0]} 10 100x500 0.1:0.3 0.2:0.3
"""

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        print(_help)
        sys.exit(0)

# First we define all the defaults, then we let the arguments overwrite
# them.

comm = MPI.COMM_WORLD
# Get rank in the `comm` communicator.

rank = comm.Get_rank()
# Get total number of MPI ranks in the `comm` communicator.
cores = comm.Get_size()


chunk_size = 10
size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

# Now grab the arguments
argv = sys.argv[1:]
if argv:
    chunk_size = int(argv.pop(0))
if argv:
    size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))

if rank == 0:
    print(f"""\
    Calculating the Mandelbrot set with these arguments:

    {chunk_size = }
    {size = }
    {xlim = }
    {ylim = }
    """)

# Convert to numpy arrays, not really needed...
size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

# Dimensions of the image


xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]


def split_problem(size,chunks):
    list_of_chunks = []
    rows = int(size[0])
    base, remainder = divmod(rows, chunks)
    low_bound = int(0)

    for i in range(chunks):
        high_bound = low_bound + int(base) + (i < int(remainder))
        list_of_chunks.append(np.array([i, low_bound, high_bound]))
        low_bound = int(high_bound)

    return list_of_chunks




if rank == 0:
    max_chunk_size = chunk_size
    chunks = int(np.ceil(size[0] / chunk_size))

    list_of_chunks = split_problem(size,chunks)
    active_workers = cores-1

    status = MPI.Status()

    full_image = np.empty((size[0],size[1]), dtype= np.float64)

    t1 = time()
    for core in range(1,cores): # Start by allocating 1 chunk per problem

        if list_of_chunks:
            problem = list_of_chunks.pop()
            comm.Send([problem, MPI.INT64_T], dest = core, tag = 0)
        else:
            comm.Send([np.array([-1, -1, -1], dtype=np.int64), MPI.INT64_T], dest = core, tag = 1)
            active_workers -= 1

    while active_workers > 0:
        incomming_chunk = np.empty((max_chunk_size,size[1]))
        comm.Recv(incomming_chunk, source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)

        worker = status.Get_source()
        start_row = status.Get_tag()

        count_floats = status.Get_count(MPI.DOUBLE)
        num_rows = count_floats // size[1]
        full_image[start_row:(start_row+num_rows),:] = incomming_chunk[:num_rows,:]

        if list_of_chunks:
            problem = list_of_chunks.pop()
            comm.Send(problem, dest = worker, tag = 0)
        else:
            comm.Isend(np.empty(3,dtype = np.int64), dest = worker, tag = 1)
            active_workers -= 1

  
    t2 = time()
    print("full image shape", full_image.shape)
    plt.rcParams.update({
        "font.size": 10,
    })

    plt.imshow(full_image.T, extent=np.concatenate([xlim, ylim]))
    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")

    # Just minimize white-space around the actual plot...
    plt.margins(0, 0)
    plt.savefig("Anton/Figure_1.png", bbox_inches="tight", pad_inches=0)
else:
    
    status = MPI.Status()

    while True:
        incomming_problem = np.empty(3, dtype = np.int64)
        comm.Recv(incomming_problem,source=0, tag=MPI.ANY_TAG, status=status)
        

        if status.tag == 1:
            break
        print(incomming_problem)
        chunk_number, low_bound, high_bound = incomming_problem
        low_bound, high_bound = int(low_bound), int(high_bound)
        
        local_image = np.zeros((int(high_bound-low_bound), size[1]), dtype = np.float64)

        print(f"Rank {rank}: Local Image shape {local_image.shape}", flush=True)


        for x in range(low_bound, high_bound):
            cx = complex(xlim[0] + x * xconst, 0)
            local_x = x - low_bound
            for y in range(size[1]):
                # process (x, y)
                c = cx + complex(0, ylim[0] + y * yconst)
                z = 0
                for i in range(100):
                    z = z*z + c
                    if np.abs(z) > 2:
                        local_image[local_x, y] = i
                        break




        comm.Send(local_image, dest = 0, tag = low_bound)
