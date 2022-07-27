#!/usr/bin/env bash
# example command to run
# bash create_container_gcp.sh -n pm-container -u shenoy -p 8001
# -n: name of the container
# -u: username (this is the name of folder you created inside outputs/ folder)
# -p: port number (this is needed if you want to start jupyter lab on a remote machine)

# get inputs
while getopts "n:u:p:" OPTION; do
	case $OPTION in
		n) name=$OPTARG;;
		u) user=$OPTARG;;
		p) port=$OPTARG;;
		*) exit 1 ;;
	esac
done

image=wadhwaniai/pest-monitoring:v2

# if [[ -z $WANDB_API_KEY ]] ; then
# echo "ERROR: set the environment variable WANDB_API_KEY"
# 	exit 0
# fi

docker run --rm -it \
	--gpus "all" \
	--shm-size 16G \
	--name "$name"_"$user"_"$port" \
	--user $(id -u):$GID \
	-v $HOME/projects/pest-monitoring-new/:/workspace/pest-monitoring-new \
	-v /etc/group:/etc/group:ro \
	-v /etc/passwd/:/etc/passwd:ro \
	-v $HOME/data:/data \
	-w /workspace/pest-monitoring-new \
	-p $port:$port \
	--ipc host \
	$image
