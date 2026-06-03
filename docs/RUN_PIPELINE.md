# Quy trình chạy lại thí nghiệm

Tài liệu này dùng cho pipeline ảnh `color_safe`, augmentation nhẹ và 4 backbone:

- `resnet50`
- `effnet_b4`
- `convnext`
- `vit`

Luôn chạy từ thư mục gốc project:

```powershell
cd D:\skin_cancer_project
$env:NO_ALBUMENTATIONS_UPDATE='1'
```

## 1. Cấu trúc script

```text
script/
  data/
    create_group_splits.py
  training/
    train_single.py
    train_ham.py
    train_bcn.py
  evaluation/
    cross_test_ham_on_bcn.py
    cross_test_bcn_on_ham.py
    ensemble_test.py
  explain/
    run_shap.py
    explain_ham.py
    explain_bcn.py
  documentation/
    build_paper_tables.py
    create_*.py
    create_*.ps1
```

## 2. Tiền xử lý ảnh

Pipeline mới mặc định là `color_safe`: bỏ Gray-World mặc định, chỉ xóa lông và tăng nhẹ kênh L trong LAB.

```powershell
.\.venv\Scripts\python.exe src\preprocessed\ham_10000_pre.py --profile color_safe --workers 4
.\.venv\Scripts\python.exe src\preprocessed\bcn_2000_pre.py --profile color_safe --workers 4
```

Output:

```text
dataset\Ham10000-color-safe-preprocessed
dataset\Bcn20000-color-safe-preprocessed
```

Ablation nếu cần:

```powershell
.\.venv\Scripts\python.exe src\preprocessed\ham_10000_pre.py --profile raw_resize --workers 4
.\.venv\Scripts\python.exe src\preprocessed\bcn_2000_pre.py --profile raw_resize --workers 4

.\.venv\Scripts\python.exe src\preprocessed\ham_10000_pre.py --profile legacy --workers 4
.\.venv\Scripts\python.exe src\preprocessed\bcn_2000_pre.py --profile legacy --workers 4
```

## 3. Tạo hoặc dùng lại split

Nếu metadata không đổi, dùng lại:

```text
dataset\metadata\group_safe\ham10000_train.csv
dataset\metadata\group_safe\ham10000_val.csv
dataset\metadata\group_safe\ham10000_test.csv
dataset\metadata\group_safe\bcn20000_train.csv
dataset\metadata\group_safe\bcn20000_val.csv
dataset\metadata\group_safe\bcn20000_test.csv
```

Nếu cần tạo lại:

```powershell
.\.venv\Scripts\python.exe script\data\create_group_splits.py `
  --dataset ham10000 `
  --source dataset\metadata\HAM_METADATA_FILE.csv `
  --output-dir dataset\metadata\group_safe

.\.venv\Scripts\python.exe script\data\create_group_splits.py `
  --dataset bcn20000 `
  --source dataset\metadata\BCN_METADATA_FILE.csv `
  --output-dir dataset\metadata\group_safe
```

Thay `HAM_METADATA_FILE.csv` và `BCN_METADATA_FILE.csv` bằng metadata gốc thật.

## 4. Train single run

Single run dùng đúng train/val/test có sẵn, không chạy 5-fold. Dùng để test nhanh pipeline mới.

HAM:

```powershell
.\.venv\Scripts\python.exe script\training\train_single.py `
  --dataset ham10000 `
  --strategy strategy3 `
  --backbone effnet_b4 `
  --augmentation-profile light
```

BCN:

```powershell
.\.venv\Scripts\python.exe script\training\train_single.py `
  --dataset bcn20000 `
  --strategy strategy3 `
  --backbone effnet_b4 `
  --augmentation-profile light
```

Chạy 4 backbone cho HAM:

```powershell
foreach ($b in @("resnet50","effnet_b4","convnext","vit")) {
  .\.venv\Scripts\python.exe script\training\train_single.py `
    --dataset ham10000 `
    --strategy strategy3 `
    --backbone $b `
    --augmentation-profile light
}
```

Chạy 4 backbone cho BCN:

```powershell
foreach ($b in @("resnet50","effnet_b4","convnext","vit")) {
  .\.venv\Scripts\python.exe script\training\train_single.py `
    --dataset bcn20000 `
    --strategy strategy3 `
    --backbone $b `
    --augmentation-profile light
}
```

Output mặc định:

```text
checkpoint_ham10000_single\single_<strategy>_<backbone>_ham10000_<augmentation>
checkpoint_bcn20000_single\single_<strategy>_<backbone>_bcn20000_<augmentation>
```

## 5. Train cross-validation 5-fold

CV nội bộ HAM:

```powershell
.\.venv\Scripts\python.exe script\training\train_ham.py `
  --strategies strategy1 strategy3 `
  --backbones resnet50 effnet_b4 convnext vit `
  --augmentation-profile light
```

CV nội bộ BCN:

```powershell
.\.venv\Scripts\python.exe script\training\train_bcn.py `
  --strategies strategy1 strategy3 `
  --backbones resnet50 effnet_b4 convnext vit `
  --augmentation-profile light
```

Chạy đủ 4 strategy:

```powershell
.\.venv\Scripts\python.exe script\training\train_ham.py `
  --strategies strategy1 strategy2 strategy3 strategy4 `
  --backbones resnet50 effnet_b4 convnext vit `
  --augmentation-profile light

.\.venv\Scripts\python.exe script\training\train_bcn.py `
  --strategies strategy1 strategy2 strategy3 strategy4 `
  --backbones resnet50 effnet_b4 convnext vit `
  --augmentation-profile light
```

Ghi chú:

- `strategy1`: image-only
- `strategy2`: concatenation
- `strategy3`: FiLM
- `strategy4`: gating
- `light`: không dùng `ColorJitter`, không `CoarseDropout`
- `none`: không augmentation, chỉ resize + normalize
- `standard`: augmentation cũ

Backbone-specific defaults đã được cấu hình trong `src/utils/experiment_runner.py`:

- ConvNeXt và ViT dùng batch nhỏ hơn và finetune thận trọng hơn.
- CLI vẫn có quyền override bằng `--batch-size`, `--epochs`, `--augmentation-profile`.

## 6. Cross-test

HAM-trained model test trên BCN:

```powershell
.\.venv\Scripts\python.exe script\evaluation\cross_test_ham_on_bcn.py `
  --checkpoint-path "D:\skin_cancer_project\checkpoint_ham10000_single\single_strategy3_effnet_b4_ham10000_light\best_strategy3_single.pt" `
  --meta-info-path "D:\skin_cancer_project\checkpoint_ham10000_single\single_strategy3_effnet_b4_ham10000_light\meta_info_single.pkl" `
  --metadata-mode strategy3 `
  --model-name tf_efficientnet_b4_ns
```

BCN-trained model test trên HAM:

```powershell
.\.venv\Scripts\python.exe script\evaluation\cross_test_bcn_on_ham.py `
  --checkpoint-path "D:\skin_cancer_project\checkpoint_bcn20000_single\single_strategy3_effnet_b4_bcn20000_light\best_strategy3_single.pt" `
  --meta-info-path "D:\skin_cancer_project\checkpoint_bcn20000_single\single_strategy3_effnet_b4_bcn20000_light\meta_info_single.pkl" `
  --metadata-mode strategy3 `
  --model-name tf_efficientnet_b4_ns
```

Với backbone khác, đổi `--model-name`:

```text
resnet50                  -> resnet50
effnet_b4                 -> tf_efficientnet_b4_ns
convnext                  -> convnext_base
vit                       -> vit_base_patch16_224
```

## 7. Sinh SHAP metadata summary

Ví dụ HAM single checkpoint:

```powershell
.\.venv\Scripts\python.exe script\explain\run_shap.py `
  --dataset ham10000 `
  --strategy strategy3 `
  --backbone effnet_b4 `
  --checkpoint-path "D:\skin_cancer_project\checkpoint_ham10000_single\single_strategy3_effnet_b4_ham10000_light\best_strategy3_single.pt" `
  --output "D:\skin_cancer_project\checkpoint_ham10000_single\shap_ham_effnet_b4.png"
```

Ví dụ BCN single checkpoint:

```powershell
.\.venv\Scripts\python.exe script\explain\run_shap.py `
  --dataset bcn20000 `
  --strategy strategy3 `
  --backbone effnet_b4 `
  --checkpoint-path "D:\skin_cancer_project\checkpoint_bcn20000_single\single_strategy3_effnet_b4_bcn20000_light\best_strategy3_single.pt" `
  --output "D:\skin_cancer_project\checkpoint_bcn20000_single\shap_bcn_effnet_b4.png"
```

Với ConvNeXt hoặc ViT, chỉ đổi `--backbone` và checkpoint path.

Lưu ý: script SHAP hiện giải thích đóng góp metadata bằng ảnh dummy zero. Đây là metadata SHAP, không phải image-pixel SHAP.

## 8. Thứ tự chạy khuyến nghị

1. Preprocess `color_safe` cho HAM và BCN.
2. Chạy `train_single.py` với `strategy1` và `strategy3` trên `effnet_b4`.
3. Nếu ổn, chạy single cho 4 backbone.
4. Chạy CV 5-fold cho cấu hình cần báo cáo chính.
5. Chạy cross-test HAM -> BCN và BCN -> HAM.
6. Sinh SHAP cho checkpoint chính.
7. Báo cáo đủ AUC, Accuracy, F1, Recall/Sensitivity, Precision, Specificity.
