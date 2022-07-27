# usage: bash jupyter.sh 8001
# starts a jupyter lab session on port 8001 (8001 should be open on docker, if running on docker)

tmux new -d -s JupLabSession
tmux send-keys -t JupLabSession.0 "cd ../; jupyter lab --no-browser --ip 0.0.0.0 --allow-root --port $1 --NotebookApp.password='sha1:3d82b019e78c:0bbbd52ffe324b07f29e9e65aeb4ad21a2c0446d'" ENTER
