# DGX Intro #2
## TL;DR
Some info in short
```
ssh <user>@login.bgd.ed.tum.de
use your TUM password
```

- https://github.com/tum-bgd/dcsample.git

```
srun hostname
srun --gpus=4 nvidia-smi
sbatch --gpus=1 jobscript.sh
```

```
# minimize cpus 128 real CPUs and memory to less than all!  
sbatch --cpus-per-gpu=16 --mem-per-gpu=100G --gpus=1 jobscript.sh 
```

# Job information as JSON
The SLURM accounting system will be used to ensure fairness when the system is under pressure. Therefore, it might be interesting to investigate the ressource consumpsion (e.g., cost) of your jobs. In order not to fight with the representation, I share a tool that I just stitched together to represent jobs information as JSON for further processing (e.g., bootstrap table might be a good thing if you know HTML a bit)
```
sacct -j  220 --format ALL -P | jq -R -s -f acct.jq
```

## Tina Tensorflow - Software Preparation 
### Step 1: Preparing Your Own System
Before starting this step, Tina has installed a Linux, NVIDIA GPU drivers, Docker, and the nvidia-container-runtime. She has already made sure that container can access GPUs. 

#### Step 1.1: Derive your container from NVIDIAs NGC Containers
Tina creates an account at [NVIDIA GPU Cloud](https://ngc.nvidia.com/). Then, in the account menu, she chooses Setup and creates an OAUTH Token to access NGC using Docker. Having generated a token,
she logs in to NGC using her local Docker installation as follows:

```
docker login nvcr.io
Username: $oauthtoken
Password: <Your Key>
```
This should end with "login succeeded."

Now, Tina is able to use NGC containers on her own GPU machine to design her personal preferred container. As said,
she is going to use NVIDIA's tensorflow container, therefore, she downloads it now. As she is interested in 
reproducibilty of her own research, she explicitly downloads a version that fits to the driver versions of the 
intended environment. At the time of writing, (early 2024), we chose the following container, but anyone following up
on Tina's track should rethink the versions based on their needs.

```
tina@tensorflow:~$ docker pull nvcr.io/nvidia/tensorflow:24.03-tf2-py3
24.03-tf2-py3: Pulling from nvidia/tensorflow
bccd10f490ab: Pulling fs layer
00ee0f9ec7c0: Pulling fs layer
4f4fb700ef54: Pulling fs layer
```
After downloading, the NVIDIA tensorflow container, Tina checks whether her scripts are possible to run with it. She already knows that she wants to use matplotlib and Jupyter notebooks, hence, she is going to define her own Docker container. Jupyter notebook is already part of the NGC container, but matplotlib is not. 

Hence, she builds her own container deriving from the selected NGC container adding her software. Note that at this stage you can add whatever **software** you want. At the moment, some people add **data** as well to containers. This is the wrong approach. Never add data to software containers!

Her container definition file is created in an empty folder and looks like
```
FROM nvcr.io/nvidia/tensorflow:24.03-tf2-py3
RUN pip3 install matplotlib
```
She builds this container and calls it Tinas Tensorflow Container ttf.
```
docker build . -t tinatensorflow/ttf
```
Now, she tries to run the container with a first notebook. She decides to train a small CNN as available from the tensorflow documentation [here](https://www.tensorflow.org/tutorials/images/classification)

Therefore, she spins up a GPU contianer on her local machine using her freshly generated image, downloads
the jupyter notebook directly from Github as below and transforms it (by running it) into another jupyter notebook containing the computational results. In another window, she is checking that the GPU is in fact being used as she observes by running `watch nvidia-smi`.

```
docker run -it --rm --name ttt-test tinatensorflow/ttf bash
wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/cnn.ipynb
 ```

### Step 1.2: Install Apptainer (formerly known as Singularity)
Docker is a tool that is helpful to develop cloud-native applications and containers. However, its design does not make it a good choice for operating large infrastructures and especially not for sharing compute ressources between untrusted users. In fact, everyone being able to instantiate Docker containers is equivalent to a root user of the system leading to cyber security risks, but as well to an organizational nightmare as everyone can start a container in the name of root. One problem is security, but a more pragmatic problem is that no admin can trace this container to a specific user or group and, therefore, managing such compute infrastructures is near to impossible unless users are well-educated, trust each other, and remain a very small group (e.g., within a single workgroup).

Singularity has emerged and was lately renamed to apptainer to alleviate the issues that Docker raised in compute: apptainer provides a container mechanism similar to Docker with the following advantages:

- Containers can run without root permissions at any time (not even during container startup)
- Containers are simple files and can be easily backed up, no layer sharing or complex dependencies are foreseen. A container is a big, self-contained file.

Tina's admins have decided to use Singularity together with the workload manager SLURM such that users can run arbitrary containers in a secure manner without accidentially or even intendedly impacting each other.

But this means, that Tina now turns her Docker container into an apptainer image. And as apptainer is very young software, she compiles the specific version she was told to use from source and installs it into her Linux system.

The following code snippet shows, what she did to her Debian/Ubuntu system. Users of other operating systems can refer to the documentation of apptainer.
```
# Ensure repositories are up-to-date
sudo apt-get update
# Install debian packages for dependencies
sudo apt-get install -y \
    build-essential \
    libseccomp-dev \
    pkg-config \
    uidmap \
    squashfs-tools \
    fakeroot \
    cryptsetup \
    tzdata \
    curl wget git
# Install Go (note that you should first check whether you are using go already)
export GOVERSION=1.20.10 OS=linux ARCH=amd64  # change this as you need
wget -O /tmp/go${GOVERSION}.${OS}-${ARCH}.tar.gz \
  https://dl.google.com/go/go${GOVERSION}.${OS}-${ARCH}.tar.gz
sudo tar -C /usr/local -xzf /tmp/go${GOVERSION}.${OS}-${ARCH}.tar.gz

echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
# Finally clone apptainer
git clone https://github.com/apptainer/apptainer.git
# With the specific version that the compute environment will be using
git checkout v1.3.0
# insid eth edirectory
cd apptainer
./mconfig -p /usr/local
cd builddir
make -j

# finally, install newest FUSE tools (not always needed, especially in the future)
sudo apt-get install -y autoconf automake libtool pkg-config libfuse3-dev zlib1g-dev
./scripts/download-dependencies
./scripts/compile-dependencies
```
Now she should have a working apptainer instance on her node. The changes to the operating environment are 
not too small (installing go, installing FUSE drivers for squashfs from source), but she is happy to have a good system now.
### Step 1.3 Playing around with apptainer and finally building the image
A new container infrastructure calls for a Hello World and, luckily, there is one:
```
$ singularity pull shub://vsoch/hello-world
INFO:    Downloading shub image
59.8MiB / 59.8MiB [===============================================================================================] 100 % 18.5 MiB/s 0s
martin@tulrbgd-g01:~/singularity$
```


- It is a good  reconfigure: always_use_nv = yes	

### Steo 1.4 Building a container image

Get a good container (we show tensorflow) from NGC. They are optimized for DGX hardware and change slowly compared to the public stream of containers for DL engines. If you need custom software, compile it through SLURM on the DGX to use the features of hard- and software. You might need to assemble a container for building including all compilers...

If you are not on the DGX (e.g., in a job), you need to configure singularity yourself. Therefore, you export SINGULARITY_DOCKER_USERNAME='$oauthtoken' and SINGULAIRTY_DOCKER_PASSWORD=<an API key you generated>

Now, you can build a TF apptainer image as follows:
```
$ singularity build tf2024.simg docker://nvcr.io/nvidia/tensorflow:24.02-tf2-py3

```
This will download, unpack and overlay all layers of the docker infrastructure. Takes a while.


# FAQ
- If you see CUDA errors like "System not Initialized", then drop an email. We need to update a kernel package 

