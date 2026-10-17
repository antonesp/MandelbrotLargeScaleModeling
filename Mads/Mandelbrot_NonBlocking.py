import sys
import time
import numpy as np
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype

def free(req: MPI.Request):
    try:
        req.free()
    except: pass

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

# print(f"""\
# Calculating the Mandelbrot set with these arguments:

# {chunk_size = }
# {size = }
# {xlim = }
# {ylim = }
# """)

# Convert to numpy arrays, not really needed...
size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

# Dimensions of the image
# image = np.zeros(size[0] * size[1])
image = np.zeros(size)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

# Create communicator
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Define mandelbrot
def mandelbrot(bounds, image):
    lw_bnd, up_bnd = bounds
    image.fill(0)
    # image = np.zeros(shape=((up_bnd - lw_bnd), size[1]), dtype=np.float64)
    for j, x in enumerate(range(lw_bnd, up_bnd)):
        cx = complex(xlim[0] + x * xconst, 0)
        for y in range(size[1]):
            # process (x, y)
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z*z + c
                if np.abs(z) > 2:
                    image[j, y] = i
                    break
    # return image

# Now send x-index to rank 0
# Get the local new dimension
total_chunks = (size[0] + chunk_size - 1) // chunk_size
no_chunks = (total_chunks - comm_rank + comm_size - 1) // comm_size

reqs = []

comm.Barrier()
if comm_rank == 0:
    t1 = time.time()

    for i in range(1, comm_size):
        rank_chunk = (total_chunks - i + comm_size - 1) // comm_size
        for j in range(rank_chunk):
            lw_bnd = (j * comm_size + i) * chunk_size
            up_bnd =  min(lw_bnd + chunk_size, size[0])
            req = comm.Irecv(image[lw_bnd:up_bnd, :], source=i, tag=j)
            reqs.append(req)

    t2 = time.time()
    for idx in range(no_chunks):
        lw_bnd = (idx * comm_size + comm_rank) * chunk_size
        up_bnd =  min(lw_bnd + chunk_size, size[0])
        mandelbrot((lw_bnd, up_bnd), image[lw_bnd:up_bnd, :])


    t3 = time.time()
    print(f"Rank {comm_rank} spent {t3 - t2:.5f} s calculating")

    MPI.Request.Waitall(reqs)
    t4 = time.time()
    print(f"Time spent running the program {t4 - t1:.5f} s")
    for req in reqs:
        free(req)
else:
    # One contiguous buffer holding all chunks for this worker (same memory as BlockingRR2D!)
    local_buffer = np.zeros(shape=(no_chunks * chunk_size, size[1]), dtype=np.float64)
    worker_reqs = []
    t1 = time.time()
    for idx in range(no_chunks):
        lw_bnd = (idx * comm_size + comm_rank) * chunk_size
        up_bnd = min(lw_bnd + chunk_size, size[0])
        rows = up_bnd - lw_bnd
        buf_slice = local_buffer[idx * chunk_size : idx * chunk_size + rows, :]
        mandelbrot((lw_bnd, up_bnd), buf_slice)
        worker_reqs.append(comm.Isend(buf_slice, dest=0, tag=idx))
    MPI.Request.Waitall(worker_reqs)
    t2 = time.time()
    print(f"Rank {comm_rank} spent {t2 - t1:.5f} s with sending")

# Visualization
import matplotlib.pyplot as plt

# Increase font-size
if comm_rank == 0:
    plt.rcParams.update({
        "font.size": 10,
    })
    plt.imshow(image.T, extent=np.concatenate([xlim, ylim]))
    plt.xlabel(rf"x / Re($p_0$)")
    plt.ylabel(rf"y / Im($p_0$)")

    # Just minimize white-space around the actual plot...
    plt.margins(0, 0)
    plt.savefig("Results/Figure_1_nonblock.png", bbox_inches="tight", pad_inches=0)
    plt.show()
