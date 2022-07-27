#!/usr/bin/env bash
# example command to run
# bash create_container.sh -g 0,1,2 -c 0-9 -n pm-container -u shenoy -p 8001
# -g: GPU numbers to use, e.g 0,1,2 means use gpu 0,1,2
# -c: container number to use, e.g. 0-9 means use container 0-9
# -n: name of the container
# -u: username (this is the name of folder you created inside outputs/ folder)
# -p: port number (this is needed if you want to start jupyter lab on a remote machine)

# group ID for `cotton` group
GID=3002

# get inputs
while getopts "g:c:n:u:p:" OPTION; do
	case $OPTION in
		g) gpu=$OPTARG;;
		c) cpulist=$OPTARG;;
		n) name=$OPTARG;;
		u) user=$OPTARG;;
		p) port=$OPTARG;;
		*) exit 1 ;;
	esac
done

image=wadhwaniai/pest-monitoring:latest

if [[ -z $WANDB_API_KEY ]] ; then
echo "ERROR: set the environment variable WANDB_API_KEY"
	exit 0
fi

# Set the cpulist to use if not specified
# If cpulist not specified, use the 10 cpus corresponding
# to the first GPU number, e.g. gpu=0, cpulist=0-9
# get the first gpu number from the string passed
if [[ -z $cpulist ]] ; then
	# get the first gpu number from the string (eg. device=0,1,2)
	gpu_num=$(echo $gpu | cut -d '=' -f 2 | cut -d ',' -f 1)
	cpulist=$(($gpu_num*10))-$(($gpu_num*10+9))
fi
echo "Using CPUs: ($cpulist) and GPU(s): $gpu"

NV_GPU=$gpu nvidia-docker run --rm -it \
	--shm-size 16G \
	--name "$name"_"$user"_"$port" \
	--user $(id -u):$GID \
	-v /etc/group:/etc/group:ro \
	-v /etc/passwd/:/etc/passwd:ro \
	-v /home/users/"$user"/projects/pest-monitoring-new/:/workspace/pest-monitoring-new \
	-v /scratchh:/scratchh \
	-v /scratchh/home/cotton-common/data:/data \
	-v /scratchh/home/"$user"/:/output \
	-v ~/.gitconfig:/etc/gitconfig \
	-w /workspace/pest-monitoring-new \
	-p $port:$port \
	--cpuset-cpus "$cpulist" \
	--ipc host \
	--net host \
	--env WANDB_DOCKER=$image \
	--env WANDB_API_KEY=$WANDB_API_KEY \
	--env TORCH_HOME="/data" \
	$image $exp
