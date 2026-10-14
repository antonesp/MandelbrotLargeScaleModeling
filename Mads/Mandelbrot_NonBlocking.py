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
def mandelbrot(image, bounds):
    lw_bnd, up_bnd = bounds
    for j, x in enumerate(range(lw_bnd, up_bnd)):
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
if (comm_rank == (comm_size - 1)):
    local_dim = size[0] // comm_size + size[0] % comm_size
else:
    local_dim = size[0] // comm_size    

reqs = []
buffs = []

# Calculate the number of chunks each process needs to calculate
no_chunks = int(np.ceil((size[0] / comm_size) / chunk_size))
base_dim = size[0] // comm_size
remainder = size[0] % comm_size

comm.Barrier()
if comm_rank == 0:
    t1 = MPI.Wtime()

    for i in range(1, comm_size):
        my_dim = base_dim + remainder if i == comm_size - 1 else base_dim
        offset = i * base_dim        
        
        for j in range(no_chunks):
            lw_bnd = offset + (j * chunk_size)
            up_bnd = min(lw_bnd + chunk_size, offset + my_dim)

            req = comm.Irecv(image[lw_bnd:up_bnd, :], source=i, tag=j)
            reqs.append(req)

    for idx in range(no_chunks):
        lw_bnd = idx * chunk_size
        up_bnd = min(lw_bnd + chunk_size, base_dim)
        image[lw_bnd:up_bnd, :] = mandelbrot(np.zeros(shape=((up_bnd - lw_bnd) * size[1])), (lw_bnd, up_bnd)).reshape(-1, size[1])
        

    MPI.Request.Waitall(reqs)

    t2 = MPI.Wtime()
    print(f"Time spent running the program {t2 - t1:.5f} s")
    for req in reqs:
        free(req)

else:
    my_dim = base_dim + remainder if comm_rank == comm_size - 1 else base_dim
    offset = comm_rank * base_dim

    local_bufferA = np.zeros(shape=(chunk_size * size[1]))
    local_bufferB = np.zeros(shape=(chunk_size * size[1]))

    reqA = None
    reqB = None

    for idx in range(no_chunks):
        lw_bnd = offset + (idx * chunk_size)
        up_bnd = min(lw_bnd + chunk_size, offset + my_dim)
        
        if idx % 2 == 0:
            count = min(len(local_bufferA), (up_bnd-lw_bnd)*size[1])
            if reqA: 
                reqA.Wait()
            local_bufferA.fill(0)
            RESULTS = mandelbrot(local_bufferA, (lw_bnd, up_bnd))[:count].reshape(-1, size[1])
            reqA = comm.Isend(RESULTS, dest=0, tag=idx)
        else:
            count = min(len(local_bufferB), (up_bnd-lw_bnd)*size[1])
            if reqB: 
                reqB.Wait()
            local_bufferB.fill(0)
            RESULTS = mandelbrot(local_bufferB, (lw_bnd, up_bnd))[:count].reshape(-1, size[1])
            reqB = comm.Isend(RESULTS, dest=0, tag=idx)
    if reqA:
        reqA.Wait()
    if reqB:
        reqB.Wait()

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
