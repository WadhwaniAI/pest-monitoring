#!/bin/bash
# example command to run
# bash launch-container.sh -d ~/data -o ~/output -r ~/pest-monitoring-new -p 8869 -g ~/.gitconfig
# -d: Directory where the input coco json files have image paths to
# -o: Directory where the outputs are directed (logs, checkpoints, predictions)
# -r: [optional] Path to the project repository
# -p: [optional; default: 8869] Port number (This helps to start jupyter lab on a remote machine)
# -g: [optional] Path to the gitconfig file you want mounted inside the container for development

Help()
{
   # Display Help
   echo "Create container for running ex[eriments using this script. It is recommended that the script be used inside a tmux session."
   echo
   echo "Syntax: bash launch-container.sh [-d|o|r|p|g|h]"
   echo "options:"
   echo -e "d \t Directory where the input coco json files have image paths to"
   echo -e "o \t Directory where the outputs are directed (logs, checkpoints, predictions)"
   echo -e "r \t [optional] Path to the project repository"
   echo -e "p \t [optional; default: 8869] Port number (This helps to start jupyter lab on a remote machine)"
   echo -e "g \t [optional] Path to the gitconfig file you want mounted inside the container for development"
   echo -e "h \t Print this Help"
   echo
   echo "Example Command:"
   echo "bash create_container.sh -d ~/data -o ~/output -r ~/pest-monitoring-new -p 8869 -g ~/.gitconfig"
}

# Get input flags
while getopts "d:o:r:p:g:h" OPTION; do
	case $OPTION in
		d) DATA_DIR=$OPTARG;;
		o) OUTPUT_DIR=$OPTARG;;
		r) PEST_REPO=$OPTARG;;
		p) PORT=$OPTARG;;
		g) GIT=$OPTARG;;
		h) Help
		exit 0;;
		*) Help
		exit 1;;
	esac
done

# Check if required flags passed
flag=0
error=""
if [[ -z $DATA_DIR ]] ; then
	flag=1
	error+="ERROR: set the flag -d for data directory \n"
fi

if [[ -z $OUTPUT_DIR ]] ; then
	flag=1
	error+="ERROR: set the flag -o for output directory \n"
fi

if [[ "$flag" -eq 1 ]] ; then
	echo -e "$error" >&2
	exit 1
fi

# Create directories
mkdir -p $DATA_DIR
mkdir -p $OUTPUT_DIR

# Parse docker required flags
user="$(id -u)"
group="$(id -g)"
image=wadhwaniai/pest-monitoring:v2

# Parse optional flags
if [[ -z $PEST_REPO ]] ; then
        PEST_REPO=$(cd "$(dirname $0)/.."; pwd)
        echo "Setting repo dir to $PEST_REPO, to change use -r flag"
fi

if [[ -z $PORT ]] ; then
	PORT=8869
	echo "Setting to default port $PORT, to change use -p flag"
fi

GITCONF=""
if [[ ! -z $GIT ]] ; then
	GITCONF=" -v $GIT:/etc/gitconfig"
fi

# Mount
MOUNTS=""
MOUNTS+=" -v $DATA_DIR:/data"
MOUNTS+=" -v $OUTPUT_DIR:/output"
MOUNTS+=" -v $PEST_REPO:/workspace/pest-monitoring-new"
MOUNTS+=" -v /etc/group:/etc/group:ro"
MOUNTS+=" -v /etc/passwd/:/etc/passwd:ro"
MOUNTS+="$GITCONF"

# Create container
nvidia-docker run --rm -it \
	--shm-size 16G \
	--name "pest_$user_$PORT" \
	--user $user:$group \
	--gpus all \
	-w /workspace/pest-monitoring-new \
	-p $PORT:$PORT \
	--ipc host \
	--net host \
	--env TORCH_HOME="/data" \
	$MOUNTS \
	$image bash
