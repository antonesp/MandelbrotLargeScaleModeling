import sys
import time
import numpy as np
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype

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
image = np.zeros(size)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

# Create communicator
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Create buffer for half of x-coordinates (NOTE: Only works for even cores)
buf = np.empty((size[0] // comm_size + size[0] % comm_size) * size[1], dtype=np.float64)

# Define mandelbrot

def mandelbrot(image, bounds):
    lwr, upp = bounds
    for j, x in enumerate(range(lwr, upp)):
        cx = complex(xlim[0] + x * xconst, 0)
        for y in range(size[1]):
            # process (x, y)
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z*z + c
                if np.abs(z) > 2:
                    image[j * size[1] + y] = i
                    break
    return image

# Now send x-index to rank 0
# Get the local new dimension
# local_dim = (size[0] - comm_rank + comm_size - 1) // comm_size
local_dim = (chunk_size - comm_rank + comm_size - 1) // comm_size

image_chunk = image.copy().reshape(chunk_size, -1)


# for _ in range(5):
comm.Barrier()
if comm_rank == 0:
    t1 = MPI.Wtime()
    rank0_buf = np.zeros(shape=(chunk_size * size[1]))
    image[::comm_size, :] = mandelbrot(rank0_buf, ()).reshape(-1, size[1])
    # for i in range(1, comm_size):
    #     status = MPI.Status()
    #     comm.Recv(buf, source=i, tag=i, status=status)
    #     n_recv = status.Get_count(from_numpy_dtype(buf.dtype))
    #     image[i::comm_size, :] = buf[:n_recv].reshape(-1, size[1]).copy()

    t2 = MPI.Wtime()
    print(f"Time spent running the program {t2 - t1:.5f} s")
# else:
#     RESULTS = mandelbrot(np.zeros(shape=(local_dim * size[1])))
#     comm.Send((RESULTS, MPI.DOUBLE_PRECISION), dest=0, tag=comm_rank)

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
    plt.savefig("Results/Figure_1_blockRR_chunk.png", bbox_inches="tight", pad_inches=0)
    plt.show()
