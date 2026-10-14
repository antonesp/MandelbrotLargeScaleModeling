#BSUB -J DynamicMandelBrot[6,8,10,12,14,16,18,20,24]
#BSUB -o outputs/10k_resolution/DynamicMandelBrot_%J_%I.out
#BSUB -e outputs/10k_resolution/DynamicMandelBrot_%J_%I.err
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -q hpcintro
#BSUB -W 00:40
#BSUB -M 8GB
#BSUB -R "select[model == XeonE5_2650v4]"



module load matplotlib/3.10.8-numpy-2.4.1-python-3.14.2
module load mpi4py/4.1.1-python-3.14.2-openmpi-5.0.9





echo "Running with $LSB_JOBINDEX cores"


mpirun -np "$LSB_JOBINDEX" python3 MandelbrotSendRecv.py 10 10000x10000
