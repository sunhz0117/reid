echo $*
gpu=${1}
method=${2}
load=${3}
save=${4}
mode=${5}
attr=${6}

if [ $mode = "train" ]
then
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM+10000)) train.py --dataset sysu --lr 0.1 --method ${method} --model_path ${save} --resume=${load} --gpu ${gpu} --attr ${attr}
elif [ $mode = "eval" ]
then
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM+10000)) test.py --dataset sysu --method ${method} --mode all --model_path ${save} --resume ${load} --gpu ${gpu} --attr ${attr}
fi

# Demo
# bash ./run.sh 2 agw sysu_agw_p4_n8_lr_0.1_seed_0.t checkpoint/demo eval 0

# Non-Local
# bash ./run.sh 2 agw '' checkpoint/xxx train 0
# bash ./run.sh 2 agw sysu_agw_p4_n8_lr_0.1_seed_0_best.t checkpoint/non_local eval 0

# bash ./run.sh 2 base '' checkpoint/xxx train 0
# bash ./run.sh 2 base sysu_base_p4_n8_lr_0.1_seed_0_best.t checkpoint/base eval 0

# Attribute
# bash ./run.sh 2 agw '' checkpoint/xxx train 39
# bash ./run.sh 2 agw sysu_agw_p4_n8_lr_0.1_seed_0_best.t checkpoint/attr eval 39

# bash ./run.sh 2 agw '' checkpoint/xxx train 22
# bash ./run.sh 2 agw sysu_agw_p4_n8_lr_0.1_seed_0_best.t checkpoint/attr22 eval 22

# bash ./run.sh 2 base '' checkpoint/xxx train 22
# bash ./run.sh 2 base sysu_base_p4_n8_lr_0.1_seed_0_best.t checkpoint/base_attr22 eval 22

# Different Loss
# bash ./run.sh 2 crosstri '' checkpoint/xxx train 22
# bash ./run.sh 2 crosstri sysu_crosstri_attr22_p4_n8_lr_0.1_seed_0_best.t checkpoint/crosstri_attr22 eval 22

# bash ./run.sh 2 rank '' checkpoint/xxx train 22
# bash ./run.sh 2 rank sysu_rank_attr22_p4_n8_lr_0.1_seed_0_best.t checkpoint/rank_attr22 eval 22

# bash ./run.sh 2 transition '' checkpoint/xxx train 22
# bash ./run.sh 2 transition sysu_transition_attr22_p4_n8_lr_0.1_seed_0_best.t checkpoint/transition_attr22 eval 22


