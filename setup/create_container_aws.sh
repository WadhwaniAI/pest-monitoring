#!/usr/bin/env bash
# example command to run
# bash create_container_aws.sh -g 0 -n pm-container -u shenoy -p 8001
# -g: gpu number
# -n: name of the container
# -u: username (this is the name of folder you created inside outputs/ folder)
# -p: port number (this is needed if you want to start jupyter lab on a remote machine)

# get inputs
while getopts "g:n:u:p:" OPTION; do
	case $OPTION in
		g) gpu=$OPTARG;;
		n) name=$OPTARG;;
		u) user=$OPTARG;;
		p) port=$OPTARG;;
		*) exit 1 ;;
	esac
done

image=wadhwaniai/pest-monitoring:v0

if [[ -z $WANDB_API_KEY ]] ; then
echo "ERROR: set the environment variable WANDB_API_KEY"
	exit 0
fi

NV_GPU=$gpu nvidia-docker run --rm -it \
	--shm-size 16G \
	--name "$gpu"_"$name"_"$user"_"$port" \
	-v $HOME/projects/pest-monitoring-new/:/workspace/pest-monitoring-new \
	-v $HOME/data:/data \
	-w /workspace/pest-monitoring-new \
	-p $port:$port \
	--ipc host \
	--env WANDB_DOCKER=$image \
	--env WANDB_API_KEY=$WANDB_API_KEY \
	$image $exp
