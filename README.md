# Quick Start Guide: Running Jobs on NCHC Supercomputers (Nano5, nVidia H100/H200)
Because the Dept of CSIE, TKU bought the GPU resources from NCHC in year 2025, to help teachers and students access this platform easily, I design this quick-start guide which introduces how to use the NCHC GPU instance by Slurm. This repository shows how to run an image classification algorithm (PyTorch) on NCHC Nano5 using Slurm and a Singularity container. 

## Quick-Start Steps
### 1. Download / Clone this Project on NCHC Nano5
```bash
git clone https://github.com/worldstar/NCHC-Slurm-Demo.git
```
### 2. Revise singularityClassification.slurm: 
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
### 3. Connect to the Nano5 Server by SSH
Please read the slides "How to use Nano5 of NCHC.pdf", particularly the Section 2.

### 4. Submit the Slurm Job: 
From the repository folder, submit the job:

```bash
sbatch singularityClassification.slurm
```

### 5. Check job status:

```bash
squeue -u $USER
```
View logs (after job starts):

```bash
cd pytorch-image-models
ls *.out *.err
less JOBNAME-JOBID.out
```

## Extra
If you are interested in the Taiwan Computing Cloud (TWCC) with Tesla V100 GPU with web interface, please read the slides "How to use TWCC-YOLOv9.pdf" in the Slides folder.