echo $*
gpu=${1}
load=${2}
save=${3}
mode=${4}
srun -J ${save} --partition=ad_ap --mpi=pmi2 --gres=gpu:${gpu} -n1 --ntasks-per-node=${gpu} python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM+10000)) test.py --gpu ${gpu} --name ${save} --batchsize 32 --which_epoch last
# if [ $mode = "train" ]
# then
#     srun -J ${save} --partition=ad_ap --mpi=pmi2 --gres=gpu:${gpu} -n1 --ntasks-per-node=${gpu} python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM+10000)) train.py --gpu ${gpu} --name ft_ResNet50 --train_all --batchsize 32
# elif [ $mode = "eval" ]
# then
#     srun -J ${save} --partition=ad_ap --mpi=pmi2 --gres=gpu:${gpu} -n1 --ntasks-per-node=${gpu} python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM+10000)) test.py --gpu ${gpu} --name ft_ResNet50 --batchsize 32 --which_epoch 59
# fi

# bash ./run.sh 1 '' ft_ResNet50 eval


