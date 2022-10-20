# Multinode NLP Training with PyTorch NGC Container

## Setup

### 1. Container image

Build from a Dockerfile and push to Docker Hub

Dockerfile sample: https://github.com/tlkh/pytorch-ngc-multinode-nlp/blob/main/Dockerfile

This Dockerfile uses a PyTorch NGC image as a base, and then:

1. Install Open MPI
2. Configure SSH
3. Install Horovod (optional)
4. Install deepspeed (required)
5. Install HuggingFace transformers library (and some other optional extras)

Use `singularity pull` to pull Docker container and convert to Singularity image

### 2. PBS Script

1. Setup resource request
2. Set up environment variables `MASTER_ADDR` and `PATH` (required for PyTorch to function correctly in some environments when the user is remapped inside the container)
3. Choose Singularity image `image`
4. `mpirun` command with required parameters

Reference for `mpirun`:

```bash
mpirun \ 
   -n NUM_OF_GPU --hostfile $PBS_NODEFILE \
   --mca pml ob1 --mca btl tcp,self,vader
   --mca btl_tcp_if_include bond0 \
   --mca btl_openib_warn_default_gid_prefix 0 \
   -bind-to none -map-by slot \
   -x NCCL_IB_GID_INDEX=3 \
   -x NCCL_CHECKS_DISABLE=1 -x NCCL_IB_DISABLE=0 \
   -x NCCL_IB_HCA=mlx5_bond_0 -x NCCL_IB_CUDA_SUPPORT=1 \
   /app/singularity/3.5.3/bin/singularity exec --nv $image \
   python SCRIPT.py
```

PBS script sample: https://github.com/tlkh/pytorch-ngc-multinode-nlp/blob/main/nlp.qsub

## NLP Training Script

### HuggingFace training

For example, see: https://github.com/tlkh/pytorch-ngc-multinode-nlp/blob/main/nlptest.py for NLP training (text classification) with HuggingFace library. When using HuggingFace's `Trainer` class, you do not need to specify additional arguments to enable multi-GPU or multi-node training.

Note that for convenience, the script will download the dataset and model weights run it is first run. If you are testing this, please run it as a single GPU job first to download the dataset and model weights. 

Note: Horovod is not used here.
