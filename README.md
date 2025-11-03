NCHC-Slurm-Demo
NCHC provides the vVidia H100 and H200 GPU. In order to access this platform quickly, I design this quick start guide which introduce how to use the NCHC GPU instance by Slurm.

How to Use Nano5 of NCHC
This repository shows how to run a PyTorch image classification job on NCHC Nano5 using Slurm and a Singularity container.

1. Download / Clone this Project on NCHC Nano5
bash
git clone https://github.com/worldstar/NCHC-Slurm-Demo.git
cd NCHC-Slurm-Demo
2. Revise singularityClassification.slurm
Edit the Slurm script to match your project/account and container:

bash
vim singularityClassification.slurm
In the file, update at least:

bash
#SBATCH --account=YOUR_PROJECT_ID   # e.g. MSTXXXX or ACDXXXX
#SBATCH --partition=dev             # or normal / normal2

CONTAINER=/work/hpc_sys/sifs/pytorch_23.11-py3.sif   # or your own .sif
python src/train.py --config configs/your_config.yaml
Save and exit:

text
:wq
3. Submit the Slurm Job
From the repository folder, submit the job:

bash
sbatch singularityClassification.slurm
Check job status:

bash
squeue -u $USER
View logs (after job starts):

bash
ls *.out *.err
less JOBNAME-JOBID.out
