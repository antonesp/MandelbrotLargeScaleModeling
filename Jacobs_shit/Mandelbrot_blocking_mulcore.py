import sys
import numpy as np
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype
import time
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_ranks = comm.Get_size()


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

# for h in ("help", "-h", "-help", "--help"):
#     if h in sys.argv:
#         print(_help)
#         sys.exit(0)

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

if rank == 0:
    t1 = time.time()

# Convert to numpy arrays, not really needed...
size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

# Dimensions of the image
xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

org_xsize = size[0]
size[0] = size[0]//num_ranks

if rank == num_ranks-1:
    img_size_x = org_xsize - size[0]*rank
    image = np.zeros((img_size_x,size[1]))
    for x in range(size[0]*rank,org_xsize):
        cx = complex(xlim[0] + x * xconst, 0)

        x_idx = x-size[0]*rank
        for y in range(size[1]):
            # process (x, y)
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z*z + c
                if np.abs(z) > 2:
                    image[x_idx, y] = i
                    break    

else:
    image = np.zeros(size)
    for x in range(size[0]*rank,size[0]*(rank+1)):
        cx = complex(xlim[0] + x * xconst, 0)

        x_idx = x-size[0]*rank
        for y in range(size[1]):
            # process (x, y)
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z*z + c
                if np.abs(z) > 2:
                    image[x_idx, y] = i
                    break

# print("rank:", rank, "image shape:",image.shape)


if rank == 0:
    images = [image]
    for i in range(1,num_ranks):
        status = MPI.Status()
        if i == num_ranks - 1:
            rows = org_xsize - size[0] * i
        else:
            rows = size[0]
        buf = np.empty((rows,size[1]), dtype=np.float64)
        comm.Recv(buf,source=i,tag=i,status=status)
        n_recv = status.Get_count(from_numpy_dtype(buf.dtype))
        images.append(buf)
    
    full_image = np.concatenate(images,axis=0)

    print(time.time()-t1)

    # # Increase font-size
    # plt.rcParams.update({
    #     "font.size": 10,
    # })
    # plt.imshow(full_image.T, extent=np.concatenate([xlim, ylim]))
    # plt.xlabel(r"x / Re(p_0)")
    # plt.ylabel(r"y / Im(p_0)")

    # # Just minimize white-space around the actual plot...
    # plt.margins(0, 0)
    # plt.savefig("Figure_1_blocking_mulcores.png", bbox_inches="tight", pad_inches=0)
    # plt.show()
else:
    message = image
    comm.Send(message,dest=0,tag=rank)


   
