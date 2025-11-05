#!/bin/bash
#SBATCH --job-name=cls-sing
#SBATCH --partition=dev
#SBATCH --account=MSTXXX      ## iService_ID 計畫 ID
## 選一種 GPU 申請方式（依叢集設定）
# 新版寫法（若支援）
##SBATCH --gpus=1
# 通用寫法（多數中心可用）
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =====[ 使用者可調參數 ]=====
CONTAINER="/work/hpc_sys/sifs/pytorch_23.11-py3.sif"
DATA_DIR="${HOME}/NCHC-Slurm-Demo/data/diabetic-retinopathy"  #We put the train, valid, and test folder here. 
SRC_DIR="${HOME}/"        # 先指定home
TIMM_DIR="${SRC_DIR}/NCHC-Slurm-Demo/pytorch-image-models"
EPOCHS=50
IMG_H=512
IMG_W=512
BATCH=64
WORKERS=${SLURM_CPUS_PER_TASK:-4}

# =====[ 基本環境 ]=====
module purge
# 如叢集提供 apptainer/singularity module，請取消註解並調整版本
# module load apptainer
# 或 module load singularity
module load singularity

# DataLoader 與 OpenMP 線程設定
export OMP_NUM_THREADS=${WORKERS}
export MKL_NUM_THREADS=${WORKERS}

# pip --user 安裝的可執行檔常在 ~/.local/bin
export PATH="${HOME}/.local/bin:${PATH}"

# =====[ 構建要在容器內執行的命令串 ]=====
CONTAINER_CMD=$(cat <<'EOS'
set -euo pipefail

# 顯示 GPU 狀態（確認 --nv 生效）
nvidia-smi || true

# 準備程式碼目錄
mkdir -p "$HOME/src"
if [ ! -d "$HOME/src/pytorch-image-models" ]; then
  git clone https://github.com/huggingface/pytorch-image-models.git "$HOME/src/pytorch-image-models"
fi

# upgrade pip to 25.3 (as requested), then deps
python -m pip --version || true
python -m pip install --user --upgrade "pip==25.3"
python -m pip --version

# 安裝/更新必要套件到使用者空間（寫入 $HOME/.local）
python -m pip install --user --upgrade pip
python -m pip install --user --upgrade wheel setuptools
# timm 建議用原始碼（含 train.py）；另外也裝到 site-packages，避免相依缺失
python -m pip install --user --upgrade timm

# 進入 timm 原始碼目錄（含 train.py）
cd "$HOME/src/pytorch-image-models"

# 開始訓練（單卡）
python -u train.py "$DATA_DIR" \
  --model convnext_base \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --workers "$WORKERS" \
  --input-size 3 "$IMG_H" "$IMG_W" \
  --pin-mem --channels-last --amp \
  --opt adamw --lr 5e-4 \
  --aa rand-m9-mstd0.5 --mixup 0.2 --cutmix 1.0 \
  --sched cosine
EOS
)

# =====[ 以 srun 啟動：Singularity/Apptainer + --nv 啟用 GPU ]=====
# --nv 會把主機的 NVIDIA 驅動/庫綁入容器，以便容器內的 PyTorch 能抓到 GPU（官方文件建議做法）
# 如果你的叢集需要額外 bind 路徑，可在 -B 補上（預設會自動掛載 $HOME）
srun singularity exec --nv \
  -B "${HOME}:${HOME}" \
  --env DATA_DIR="${DATA_DIR}",EPOCHS="${EPOCHS}",IMG_H="${IMG_H}",IMG_W="${IMG_W}",BATCH="${BATCH}",WORKERS="${WORKERS}" \
  "${CONTAINER}" \
  bash -lc "$CONTAINER_CMD"
