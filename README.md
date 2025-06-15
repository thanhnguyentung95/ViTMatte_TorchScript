# ViTMatte Torchscript

This repository provides tools to convert **ViTMatte**—a high-quality portrait matting model based on Vision Transformers—into **TorchScript** format. The exported model can then be used directly for deployment or further converted to other formats such as **ONNX** for broader platform compatibility and optimization.

---

## 🚀 Step 1: Environment Setup (Python 3.10.12)

### 1.1 Create and activate virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### 1.2 Set CUDA path for GPU acceleration

```bash
export CUDA_HOME=/usr/local/cuda-11.7
```

> Make sure your system has compatible NVIDIA drivers and CUDA version.

### 1.3 Upgrade build tools

```bash
python3 -m pip install --upgrade setuptools wheel
```

### 1.4 Install Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 1.5 Install Detectron2

ViTMatte requires [Detectron2](https://github.com/facebookresearch/detectron2) from Facebook AI Research:

```bash
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

---

## 📦 Step 2: Download Pretrained Checkpoints

| Model                       | SAD   | MSE | Grad | Conn  | Checkpoints                                                                                        |
| --------------------------- | ----- | --- | ---- | ----- | -------------------------------------------------------------------------------------------------- |
| ViTMatte-B Composition-1k   | 20.33 | 3.0 | 6.74 | 14.78 | [Google Drive](https://drive.google.com/file/d/1mOO5MMU4kwhNX96AlfpwjAoMM4V5w3k-/view?usp=sharing) |
| ViTMatte-B Distinctions-646 | 17.05 | 1.5 | 7.03 | 12.95 | [Google Drive](https://drive.google.com/file/d/1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW/view?usp=sharing) |

> Download and place the checkpoints in a directory like `pretrained/` for use in the next step.

---

## 🧪 Step 3: Export TorchScript Model

Once the checkpoint is in place, export the TorchScript version of the model:

```bash
python3 export_torchscript.py --checkpoint-dir pretrained/vitmatte-b.pth
```


## Citation
```
@article{yao2024vitmatte,
  title={ViTMatte: Boosting image matting with pre-trained plain vision transformers},
  author={Yao, Jingfeng and Wang, Xinggang and Yang, Shusheng and Wang, Baoyuan},
  journal={Information Fusion},
  volume={103},
  pages={102091},
  year={2024},
  publisher={Elsevier}
}
```
