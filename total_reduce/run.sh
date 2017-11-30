#!/bin/bash
#./clear_shm.sh

#. ./env.sh
echo set environment variable
source ./common.sh
echo end set environment variable

# generate host list
~/bin/hostlist.sh >./mpd.$HOSTNAME.txt
cat mpd.$HOSTNAME.txt

# SKX OPA uncomment the following lines
if [ "$1" == "opa" ]
then
    echo Using OPA
    export I_MPI_FABRICS=tmi # tcp
    export I_MPI_TMI_PROVIDER=psm2
else
    # SKX 10GbE uncomment the following lines
    export I_MPI_FABRICS=tcp
    export I_MPI_TCP_NETMASK=enp134s0f0
fi


# uncomment the following line to enable debugging
#export I_MPI_DEBUG=1000

#export I_MPI_COLLECTIVE_DEFAULTS=tmi

# 1: Recursive doubling
# 2: Rabenseifner's
# 3: Reduce + Bcast
# 4: Topology aware Reduce + Bcast
# 5: Binomial gather + scatter
# 6: Topology aware binominal gather + scatter
# 7: Shumilin's ring
# 8: Ring
# 9: Knomial
# 10: Topology aware SHM-based flat
# 11: Topology aware SHM-based Knomial
# 12: Topology aware SHM-based Knary
# 13: Rabenseifner's allreduce for non-inplace case
# 14: Ring allreduce for non-inplace case
# 15: N-reduce allreduce optimized
export I_MPI_ADJUST_ALLREDUCE=8

# 1: Recursive doubling
# 2: Rabenseifner’s
# 3: Reduce + Bcast
# 4: Ring (patarasuk)
# 5: Knomial
# 6: Binomial
# 7: N-reduce
# 8: N-reduce accumulate
export I_MPI_ADJUST_IALLREDUCE=4
export I_MPI_DEBUG=0

# 1: Bruck's
# 2: Isend/Irecv + waitall
# 3: Pair wise exchange
# 4: Plum's
# export I_MPI_ADJUST_ALLTOALL=2

# 1: Recursive doubling
# 2: Bruck's
# 3: Ring
# 4: Topology aware Gatherv + Bcast
# 5: Knomial
# export I_MPI_ADJUST_ALLGATHER=3

#export I_MPI_SPIN_COUNT=1
#export I_MPI_THREAD_YIELD=0

echo begin mpirun
squeue -w $HOSTNAME --format="%D"
#mpirun -l -n `squeue -w $HOSTNAME --format="%D"|tail -n 1` -ppn 1 -machinefile ./mpd.$HOSTNAME.txt ./reduce_avg_intel  99999999

export MLSL_NUM_SERVERS=2
export MLSL_MSG_PRIORITY=0
#export MLSL_LOG_LEVEL=6
export MLSL_ROOT=~/total-reduce.mlt/external/mlsl/l_mlsl_2018.0.003
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/total-reduce.mlt/external/mlsl/l_mlsl_2018.0.003/intel64/lib

#mpirun -l -n `squeue -w $HOSTNAME --format="%D"|tail -n 1` -ppn 1 -machinefile ./mpd.$HOSTNAME.txt ./workload_mlsl_inception_v3
mpirun -l -n `squeue -w $HOSTNAME --format="%D"|tail -n 1` -ppn 1 -machinefile ./mpd.$HOSTNAME.txt ./workload_inception_v3

#mpirun -l -n `squeue -w $HOSTNAME --format="%D"|tail -n 1` -ppn 1 -machinefile ./mpd.$HOSTNAME.txt ./workload_mlsl_res50
#mpirun -gdb -l -n `squeue -w $HOSTNAME --format="%D"|tail -n 1` -ppn 1 -machinefile ./mpd.$HOSTNAME.txt ./workload_res50
mpirun -l -n `squeue -w $HOSTNAME --format="%D"|tail -n 1` -ppn 1 -machinefile ./mpd.$HOSTNAME.txt ./workload_res50

#mpirun -l -n `squeue -w $HOSTNAME --format="%D"|tail -n 1` -ppn 1 -machinefile ./mpd.$HOSTNAME.txt ./workload_mlsl_vgg16
mpirun -l -n `squeue -w $HOSTNAME --format="%D"|tail -n 1` -ppn 1 -machinefile ./mpd.$HOSTNAME.txt ./workload_vgg16

echo end mpirun

