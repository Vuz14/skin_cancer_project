import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler  # Tăng tốc cho RTX 3050
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import time
import csv
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from colorama import Fore, Style, init
from tabulate import tabulate

# XAI Libs
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Fix lỗi màu sắc trên Windows CMD
init(autoreset=True)

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG (PC CONFIG)
# ==========================================
# >>> CHỌN DATASET: 'HAM10k' HOẶC 'BCN20k' <<<
CURRENT_DATASET = 'HAM10k'

CONFIG = {
    "SEED": 42,
    "BATCH_SIZE": 16,  # 16 là an toàn cho RTX 3050 (4-6GB VRAM) với ảnh 384px
    "LR": 1e-4,
    "EPOCHS": 10,
    "IMG_SIZE": 384,  # Độ phân giải cao chuẩn bài báo
    "T_SAMPLES": 10,  # Số lần chạy MC Dropout
    "NUM_CLASSES": 2,
    "LOG_FILE": "training_log_pc.csv",
    "MODEL_PATH": "best_melanomanet_pc.pth",
    "NUM_WORKERS": 4  # Tận dụng i7-12700F (có thể giảm xuống 0 nếu lỗi trên Windows)
}

# Đường dẫn tương đối (Relative Paths) cho PC
DATASET_INFO = {
    'HAM10k': {
        'IMG_DIR': './Dataset/HAM10k/',
        'CSV_PATH': './Dataset/HAM10000.csv',
        'COL_ID': 'image_id', 'COL_LABEL': 'dx'
    },
    'BCN20k': {
        'IMG_DIR': './Dataset/BCN20k/',
        'CSV_PATH': './Dataset/BCN20k.csv',
        'COL_ID': 'isic_id', 'COL_LABEL': 'diagnosis_1'
    }
}
CUR_INFO = DATASET_INFO[CURRENT_DATASET]

# Tự động nhận diện GPU NVIDIA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"{Fore.GREEN}=== SYSTEM READY ON: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ==={Style.RESET_ALL}")


# ==========================================
# 2. DATASET & PREPROCESSING
# ==========================================
class SkinLesionDataset(Dataset):
    def __init__(self, dataset_name, transform=None):
        self.dataset_name = dataset_name
        self.transform = transform
        self.info = DATASET_INFO[dataset_name]

        if not os.path.exists(self.info['CSV_PATH']):
            print(f"{Fore.RED}[LỖI] Không tìm thấy file CSV tại: {self.info['CSV_PATH']}")
            print(f"Hãy kiểm tra lại cấu trúc thư mục!{Style.RESET_ALL}")
            exit()

        raw_df = pd.read_csv(self.info['CSV_PATH'])

        # Logic Mapping
        if dataset_name == 'HAM10k':
            binary_map = {'nv': 0, 'bkl': 0, 'df': 0, 'vasc': 0, 'mel': 1, 'bcc': 1, 'akiec': 1}
            # Lọc chỉ lấy các dòng có nhãn hợp lệ
            raw_df = raw_df[raw_df[self.info['COL_LABEL']].isin(binary_map.keys())]
            raw_df['binary_label'] = raw_df[self.info['COL_LABEL']].map(binary_map)

        elif dataset_name == 'BCN20k':
            def parse_bcn(val):
                s = str(val).lower()
                if 'malignant' in s or 'melanoma' in s or 'carcinoma' in s: return 1
                if 'benign' in s or 'nevus' in s or 'keratosis' in s: return 0
                return -1

            raw_df['binary_label'] = raw_df[self.info['COL_LABEL']].apply(parse_bcn)
            raw_df = raw_df[raw_df['binary_label'] != -1]

        self.df = raw_df.reset_index(drop=True)
        print(f"[{dataset_name}] Đã tải {len(self.df)} ảnh.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx][self.info['COL_ID']]
        # Xử lý đuôi file linh hoạt
        img_name = f"{img_id}" if str(img_id).endswith('.jpg') else f"{img_id}.jpg"
        img_path = os.path.join(self.info['IMG_DIR'], img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Tạo ảnh đen nếu lỗi, tránh dừng training
            image = Image.new('RGB', (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), (0, 0, 0))

        label = int(self.df.iloc[idx]['binary_label'])
        if self.transform: image = self.transform(image)
        return image, label


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20), transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class MelanomaNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Tải trọng số EfficientNet V2-M từ internet (lần đầu sẽ hơi lâu)
        self.base = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)

        # Sửa Classifier để hỗ trợ MC Dropout
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.base.classifier[1].in_features, num_classes)
        )

    def forward(self, x): return self.base(x)

    def predict_decomposed_uncertainty(self, x, T=10):
        self.train()  # Bật Dropout
        probs_list = []
        with torch.no_grad():
            for _ in range(T):
                probs_list.append(torch.softmax(self.forward(x), dim=1))

        probs = torch.stack(probs_list)
        mean_p = probs.mean(dim=0)
        pred_u = -torch.sum(mean_p * torch.log(mean_p + 1e-8), dim=1)
        epi_u = torch.var(probs, dim=0).mean(dim=1)
        ale_u = (-torch.sum(probs * torch.log(probs + 1e-8), dim=2)).mean(dim=0)
        return mean_p, pred_u, epi_u, ale_u


# ==========================================
# 4. ABCDE & METRICS UTILS
# ==========================================
class ABCDEAnalyzer:
    @staticmethod
    def analyze(img_np):
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(mask > 0) > np.sum(mask == 0): mask = cv2.bitwise_not(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        res = {}
        res['A'] = np.sum(cv2.absdiff(mask, cv2.flip(mask, 1))) / (mask.sum() + 1e-8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res['B'] = 0.0
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            # Compactness formula
            res['B'] = (cv2.arcLength(c, True) ** 2) / (4 * np.pi * cv2.contourArea(c) + 1e-8)

        pixels = img_np[mask > 0].reshape(-1, 3)
        res['C'] = 1
        if len(pixels) > 50:
            kmeans = KMeans(n_clusters=min(6, len(pixels)), n_init=5).fit(pixels)
            res['C'] = np.unique(kmeans.labels_).size

        x, y, w, h = cv2.boundingRect(mask)
        res['D'] = max(w, h)

        # Risk Score Calculation
        risk = 0
        if res['A'] > 0.3: risk += 1
        if res['B'] > 1.4: risk += 1
        if res['C'] > 3: risk += 1
        if res['D'] > 114: risk += 1
        res['Risk_Score'] = risk

        return res, mask

    @staticmethod
    def calculate_alignment(heatmap, mask):
        if heatmap.shape != mask.shape:
            heatmap = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]))
        mask_bin = (mask > 0).astype(float)
        return np.sum(heatmap * mask_bin) / (np.sum(heatmap) + 1e-8)


def compute_metrics_sklearn(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) > 1 else 0.0
    except:
        auc = 0.0
    return acc, prec, rec, f1, auc


def save_log(epoch, train_loss, val_m, time_s):
    exists = os.path.isfile(CONFIG["LOG_FILE"])
    with open(CONFIG["LOG_FILE"], 'a', newline='') as f:
        w = csv.writer(f)
        if not exists: w.writerow(['Epoch', 'Loss', 'Acc', 'Prec', 'Rec', 'F1', 'AUC', 'Time'])
        w.writerow([epoch, f"{train_loss:.4f}", *[f"{v:.4f}" for v in val_m], f"{time_s:.1f}"])


# ==========================================
# 5. TRAINING ENGINE (PC OPTIMIZED)
# ==========================================
def train_melanomanet():
    print(f"\n{Fore.MAGENTA}=== BẮT ĐẦU HUẤN LUYỆN TRÊN PC (AMP ENABLED) ==={Style.RESET_ALL}")

    # Init Dataset
    try:
        ds = SkinLesionDataset(CURRENT_DATASET)
    except Exception as e:
        print(e)
        return None, None

    L = len(ds)
    train_sz, val_sz = int(0.7 * L), int(0.15 * L)
    test_sz = L - train_sz - val_sz
    train_set, val_set, test_set = random_split(ds, [train_sz, val_sz, test_sz],
                                                generator=torch.Generator().manual_seed(CONFIG["SEED"]))

    train_set.dataset.transform = data_transforms['train']
    val_set.dataset.transform = data_transforms['val']

    # DataLoader: persistent_workers=True giúp CPU không phải khởi động lại workers mỗi epoch
    train_loader = DataLoader(train_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True,
                              persistent_workers=(CONFIG["NUM_WORKERS"] > 0))
    val_loader = DataLoader(val_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True,
                            persistent_workers=(CONFIG["NUM_WORKERS"] > 0))

    # Model Setup
    model = MelanomaNetModel(CONFIG["NUM_CLASSES"]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Mixed Precision Scaler (Quan trọng cho RTX 3050)
    scaler = GradScaler()

    best_f1 = 0.0

    for epoch in range(CONFIG["EPOCHS"]):
        start = time.time()
        model.train()
        losses = []

        # Thanh tiến trình
        bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['EPOCHS']}", leave=False)

        for x, y in bar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            # Chạy với Mixed Precision (FP16)
            with autocast():
                out = model(x)
                loss = criterion(out, y)

            # Backward với Scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        all_pred, all_true, all_prob = [], [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                prob = torch.softmax(out, dim=1)
                pred = torch.argmax(out, dim=1)
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(y.cpu().numpy())
                all_prob.extend(prob.cpu().numpy())

        # Metrics
        met = compute_metrics_sklearn(all_true, all_pred, np.array(all_prob))
        dur = time.time() - start

        # Báo cáo
        headers = ["Metric", "Value"]
        data = [["Loss", np.mean(losses)], ["Acc", met[0]], ["Prec", met[1]], ["Rec", met[2]], ["F1", met[3]],
                ["AUC", met[4]]]

        print(f"\nEpoch {epoch + 1} Report ({dur:.1f}s):")
        print(tabulate(data, headers, floatfmt=".4f", tablefmt="simple"))
        save_log(epoch + 1, np.mean(losses), met, dur)

        # Lưu model tốt nhất
        if met[3] > best_f1:
            best_f1 = met[3]
            torch.save(model.state_dict(), CONFIG["MODEL_PATH"])
            print(f"{Fore.GREEN}>> New Best Model Saved!{Style.RESET_ALL}")

    return model, test_set


# ==========================================
# 6. DEMO FUNCTION
# ==========================================
def run_full_demo(model, test_set):
    print(f"\n{Fore.YELLOW}=== CHẠY DEMO CHẨN ĐOÁN (XAI) ==={Style.RESET_ALL}")

    # Vẽ biểu đồ lịch sử
    if os.path.exists(CONFIG["LOG_FILE"]):
        try:
            df = pd.read_csv(CONFIG["LOG_FILE"])
            plt.figure(figsize=(12, 4))
            plt.subplot(121);
            plt.plot(df['Epoch'], df['Loss']);
            plt.title("Training Loss")
            plt.subplot(122);
            plt.plot(df['Epoch'], df['F1']);
            plt.title("Validation F1-Score")
            plt.show()
        except:
            pass

    model.eval()
    idx = random.randint(0, len(test_set) - 1)
    img_t, label = test_set[idx]
    img_in = img_t.unsqueeze(0).to(DEVICE)

    # 1. Uncertainty Analysis
    p_bar, p_u, e_u, a_u = model.predict_decomposed_uncertainty(img_in)
    pred = torch.argmax(p_bar).item()

    # 2. GradCAM++ (Chạy trên CPU để tiết kiệm VRAM cho việc vẽ)
    model.cpu()
    cam = GradCAMPlusPlus(model=model, target_layers=[model.base.features[-1]])
    map = cam(input_tensor=img_t.unsqueeze(0), targets=[ClassifierOutputTarget(pred)])[0, :]
    model.to(DEVICE)  # Đưa lại về GPU

    # 3. ABCDE Analysis
    # Denormalize ảnh để hiển thị đúng màu
    orig = (img_t.permute(1, 2, 0).cpu().numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    orig = np.clip(orig, 0, 255).astype(np.uint8)
    abcde, mask = ABCDEAnalyzer.analyze(orig)
    alignment = ABCDEAnalyzer.calculate_alignment(map, mask)

    # In kết quả
    lbls = ["LÀNH TÍNH", "ÁC TÍNH"]
    print(f"\nChẩn đoán: {lbls[pred]} (Thực tế: {lbls[label]}) | Độ tin cậy: {p_bar[0][pred]:.2%}")
    print(f"Chỉ số Uncertainty: Total={p_u.item():.3f} (Epi={e_u.item():.3f}, Ale={a_u.item():.3f})")
    print(f"Phân tích ABCDE: A={abcde['A']:.2f}, B={abcde['B']:.2f}, C={abcde['C']}, D={abcde['D']}px")
    print(f"Điểm rủi ro: {abcde['Risk_Score']}/4 | Độ khớp AI-Mask: {alignment:.2f}")

    # Hiển thị hình ảnh
    viz = show_cam_on_image(orig / 255.0, map, use_rgb=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1);
    plt.imshow(orig);
    plt.title("Ảnh gốc")
    plt.subplot(1, 3, 2);
    plt.imshow(mask, cmap='gray');
    plt.title("Vùng bệnh (Mask)")
    plt.subplot(1, 3, 3);
    plt.imshow(viz);
    plt.title("Vùng AI chú ý (GradCAM++)")
    plt.show()


# ==========================================
# 7. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Dòng này cực kỳ quan trọng trên Windows để chạy đa luồng
    if os.path.exists(CONFIG["LOG_FILE"]): os.remove(CONFIG["LOG_FILE"])

    model, test_data = train_melanomanet()

    if model:
        run_full_demo(model, test_data)