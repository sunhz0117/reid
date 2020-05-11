echo $*
gpu=${1}
load=${2}
save=${3}
other=${4}
other2=${5}

CUDA_VISIBLE_DEVICES=${gpu} python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM+10000)) main.py --pre_train=${load} --save=${save} --random_erasing --re_rank --amsgrad ${other} ${other2}

# Demo
# bash ./run.sh 1 experiment/demo/model/model_best.pt demo --test_only

# MGN
# bash ./run.sh 1 '' xxx
# bash ./run.sh 1 experiment/mgn/model/model_best.pt mgn --test_only

# MGN + Attribute
# bash ./run.sh 1 experiment/mgn_attr/model/model_best.pt mgn_attr --test_only
