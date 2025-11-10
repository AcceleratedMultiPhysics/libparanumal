#!/bin/bash
#SBATCH -p barbun-cuda      # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A aaygun           # Kullanici adi
#SBATCH -J vector_scaling        # Gonderilen isin ismi
#SBATCH -o TGV_nostab_node8_64x4.out    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:2        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 8                # Gorev kac node'da calisacak?
#SBATCH -n 16                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 20  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=00:30:00      # Sure siniri koyun.

export CUDA_DIR='/arf/sw/cuda/12.4'
export PATH=$PATH:${CUDA_DIR}/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_DIR}/lib64

module purge

module load comp/cmake/3.31.1
module load lib/cuda/12.4
module load lib/openmpi/5.0.4
module load comp/gcc/12.3.0
module load lib/openblas/0.3.29


echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"


#BNS_DIR=/arf/home/aaygun/meshDeform_libp/libparanumal/solvers/bns
CNS_DIR=/arf/home/aaygun/aero/libparanumal/solvers/cns
#mpiexec -np 1 --map-by slot:PE=1 ./cnsMain setups/setupHex3D.rc
cd $CNS_DIR
mpirun ./cnsMain setups/setupTGV3D.rc
exit
