#NCHC-Slurm-Demo
NCHC provides the nVidia H100 and H200 GPU. Because the Dept of CSIE, TKU bought the GPU resource from NCHC in year 2025, to help teachers and students access this platform with a tutorial, I design this quick start guide which introduces how to use the NCHC GPU instance by Slurm. This repository shows how to run a PyTorch image classification job on NCHC Nano5 using Slurm and a Singularity container. For the details, please read the slides "How to use Nano5 of NCHC.pdf"

##1. Download / Clone this Project on NCHC Nano5
```bash
git clone https://github.com/worldstar/NCHC-Slurm-Demo.git
```
##2. Revise singularityClassification.slurm: 
Edit the Slurm script to match your project/account and container:

```bash
cd NCHC-Slurm-Demo
vim singularityClassification.slurm
```
In this file, please update YOUR_PROJECT_ID:

```bash
#SBATCH --account=YOUR_PROJECT_ID   # e.g. MSTXXXX or ACDXXXX

```
Save and exit:

```bash
:wq
```
##3. Submit the Slurm Job: 
From the repository folder, submit the job:

```bash
sbatch singularityClassification.slurm
```

##4. Check job status:

```bash
squeue -u $USER
```
View logs (after job starts):

```bash
cd pytorch-image-models
ls *.out *.err
less JOBNAME-JOBID.out
```