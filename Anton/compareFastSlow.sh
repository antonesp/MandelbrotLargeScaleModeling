#BSUB -J compareFastSlow
#BSUB -o FastNFlow_%J.out
#BSUB -e FastNFlow_%J.err
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -q hpcintro
#BSUB -W 01:00
#BSUB -M 2GB
#BSUB -R "select[model == XeonE5_2650v4]"



module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9





echo "Running with 24 cores"

echo "Running Slow"
mpirun -np 24 python3 MandelbrotSendRecv.py 10 10000x10000
mpirun -np 24 python3 MandelbrotSendRecv.py 10 10000x10000
mpirun -np 24 python3 MandelbrotSendRecv.py 10 10000x10000

echo "Runnig Fast"
mpirun -np 24 python3 MandelbrotSendRecvFast.py 10 10000x10000
mpirun -np 24 python3 MandelbrotSendRecvFast.py 10 10000x10000
mpirun -np 24 python3 MandelbrotSendRecvFast.py 10 10000x10000