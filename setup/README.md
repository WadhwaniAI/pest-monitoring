## Setup
#### 1. Clone the Repository
Clone the repository and we will be mounting it in our docker container too.

```bash
cd ~/
mkdir projects; cd projects;
git clone https://github.com/WadhwaniAI/pest-monitoring-new.git
```

#### 2. Setting up Docker Image
If the docker image is not present,

If you have access to Wadhwani AI's docker hub
```bash
cd ~/projects/pest-monitoring-new/setup/

docker pull wadhwaniai/pest-monitoring:latest
```

OR if you want to build the docker image from the Dockerfile
```bash
bash build.sh
```

#### 3. Creating a Container
After this, create a container,
```bash
cd ~/projects/pest-monitoring-new/setup/

# for a GPU machine (For Wadhwani Users)
bash create_container.sh -g 0 -c 1-10 -n pm-container -u $USER -p 8001

>>> Explanation
-g: GPU number, pass -1 for a non-GPU machine
-c: CPU list, to be assigned
-n: name of the container
-e: path to the folder where data and outputs are to be stored
-u: username (this is the name of folder you created inside outputs/ folder)
-p: port number (this is needed if you want to start jupyter lab on a remote machine)

# for a GPU machine (For Non-Wadhwani Users)
bash launch-container.sh -d ~/data -o ~/output -r ~/pest-monitoring-new -p 8869 -g ~/.gitconfig

>>> Explanation
-d: Directory where the input coco json files have image paths to
-o: Directory where the outputs are directed (logs, checkpoints, predictions)
-r: [optional] Path to the project repository
-p: [optional; default: 8869] Port number (This helps to start jupyter lab on a remote machine)
-g: [optional] Path to the gitconfig file you want mounted inside the container for development
```

#### 4. Jupyter Notebook
To start a jupyter notebook,
```bash
cd ~/projects/pest-monitoring-new/setup/

# for a GPU machine (For Wadhwani Users)
bash jupyter.sh 8001

>>> Explanation
NOTE: Use the same port as the one used to start the container
```
