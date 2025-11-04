# eccDNAMamba

A reproducible Python 3.9 environment using Conda and pip-based requirements.

## Environment Setup

### 1. Create the Conda environment with Python 3.9

```bash
conda create -n eccDNAMamba python=3.9
conda activate eccDNAMamba
```

### 2. Install PyTorch 2.1.0 with CUDA 11.8

```bash
pip install torch==2.2.0+cu121 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install nvcc via Conda

```bash
conda install -c "nvidia/label/cuda-12.1.105" cuda-nvcc
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt --no-deps
```
```bash
pip install datasets numpy==1.26.3 pyfaidx==0.8.1.3 mamba_ssm==2.2.4 causal_conv1d==1.5.0.post8 --no-cache-dir --no-build-isolation
```

### 5. Pre-training
```bash
python main.py
```

### 5. Quick Start
```bash
python cancer_eccdna_prediction_ultra_long.py
```

