PORT=$(($RANDOM % 1000 + 10000))

dataset=cifar10 # cifar100 or cifar10
imb=0.02     #0.01 0.02 or 0.1
lr=0.3
batch_size=256
N_GPU=1
epochs=200        # 200 or 400
data=/home/xiongshihao/code/data # replace with your own path
root_log=saved
test=$4 #/home/xiongshihao/code/proco/ckp # checkpoint path (default: None)

mark=dataset_${dataset}_imb${imb}_lr${lr}_batch_size${batch_size}_N_GPU${N_GPU}_epochs${epochs}
output_dir=./${root_log}/${mark}

echo ${mark}
mkdir -p ${output_dir}
cp -r ./ProCo_cifar ${output_dir}
cp ./sh/ProCo_CIFAR.sh ${output_dir}

#torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} \
python -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} \
	--nnodes=1 --nproc_per_node="${N_GPU}" \
	ProCo_cifar/main.py \
	--workers 8 \
	--dataset ${dataset} \
	--imb_factor ${imb} \
	--lr ${lr} \
	-b ${batch_size} \
	--epochs ${epochs} \
	--data ${data} \
	--root_log ${root_log} \
	--mark "${mark}" \
	--test ${test}
