import torch
import cv2
import shap
import pytorch_grad_cam
import timm
import pandas as pd


print(f"✅ Torch Version: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"✅ OpenCV Version: {cv2.__version__}")
print(f"✅ SHAP Version: {shap.__version__}")
print(f"✅ TIMM Version: {timm.__version__}")
