import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
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


# We need to split the set into cores pieces
# Calculate the piece size

if rank < cores -1 :
    low_bound = rank * int(np.floor(size[0] / cores))
    high_bound = (rank+1) * int(np.floor(size[0] / cores))
    x_dim = int(high_bound-low_bound)
else:
    low_bound = rank * int(np.floor(size[0] / cores))
    high_bound = size[0]
    x_dim = int(high_bound-low_bound)

local_image = np.zeros((x_dim, size[1]), dtype = np.float32)

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


send_buf = local_image.astype(np.float32)

if rank == 0:
    full_image = []
    full_image.append(send_buf)
    for i in range(1, cores):
        buf = np.empty(size, dtype=np.float32)
        comm.Recv(buf, source=i, tag=i)
        full_image.append(buf)
else:
    status = MPI.Status()
    comm.Send(send_buf, dest=0, tag=rank)
    

if rank == 0:
    full_image = np.array(full_image)
    print("Full image shape", full_image.shape)
    # Increase font-size
    plt.rcParams.update({
        "font.size": 10,
    })
    plt.imshow(full_image.T, extent=np.concatenate([xlim, ylim]))
    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")

    # Just minimize white-space around the actual plot...
    plt.margins(0, 0)
    plt.savefig("Figure_1.png", bbox_inches="tight", pad_inches=0)
