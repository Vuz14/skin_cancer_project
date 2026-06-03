# Multimodal Skin Cancer Diagnosis

Đây là dự án nghiên cứu mô hình AI hỗ trợ phân loại nguy cơ ung thư da từ ảnh dermoscopy kết hợp metadata lâm sàng trước chẩn đoán.

Dự án tập trung vào ba hướng chính:

- Xây dựng pipeline tiền xử lý ảnh dermoscopy theo hướng hạn chế làm biến dạng màu.
- Huấn luyện và so sánh các mô hình image-only và multimodal fusion.
- Đánh giá mô hình bằng lesion-level split, cross-validation, cross-dataset testing và phân tích giải thích bằng SHAP/Grad-CAM.

## Bài toán

Input của hệ thống gồm:

- ảnh dermoscopy
- metadata lâm sàng trước chẩn đoán, ví dụ tuổi, giới tính, vị trí tổn thương

Output là dự đoán nhị phân:

```text
0 = lành tính
1 = ác tính
```

## Dữ liệu

Project làm việc với hai bộ dữ liệu chính:

```text
HAM10000
BCN20000
```

Các split được tổ chức theo hướng lesion-level để hạn chế việc ảnh của cùng một tổn thương xuất hiện đồng thời ở train/validation/test.

## Mô hình

Project hỗ trợ bốn backbone:

```text
ResNet50
EfficientNet-B4
ConvNeXt
Vision Transformer
```

Các chiến lược fusion:

```text
Strategy 1: Image-only baseline
Strategy 2: Concatenation
Strategy 3: FiLM
Strategy 4: Gating
```

Mục tiêu là so sánh liệu metadata lâm sàng có giúp mô hình cải thiện khả năng phân loại so với chỉ dùng ảnh hay không.

## Cấu trúc thư mục

```text
src/
  data_logic/
    Dataset classes và image transforms.

  models/
    Backbone wrappers và fusion heads.

  preprocessed/
    Code tiền xử lý ảnh HAM10000 và BCN20000.

  utils/
    Training loop, loss, scheduler, metadata encoder utilities.

  evaluate/
    Script hỗ trợ vẽ biểu đồ và kiểm định kết quả.

script/
  data/
    Tạo train/validation/test split theo lesion_id.

  training/
    Script huấn luyện single run và 5-fold cross-validation.

  evaluation/
    Cross-test giữa HAM10000 và BCN20000, ensemble evaluation.

  explain/
    SHAP và Grad-CAM/XAI analysis.

  documentation/
    Script sinh bảng, tài liệu, sơ đồ và nội dung báo cáo.

docs/
  Tài liệu hướng dẫn quy trình chạy lại thí nghiệm.

dataset/
  Dữ liệu ảnh, metadata và các phiên bản ảnh đã tiền xử lý.

checkpoint_ham10000/
checkpoint_bcn20000/
  Checkpoint và kết quả huấn luyện theo từng dataset.

deliverables/
  Tài liệu, hình ảnh và sản phẩm báo cáo đã sinh ra.
```

## Tiền xử lý ảnh

Pipeline mới ưu tiên profile `color_safe`.

Khác với pipeline cũ, profile này hạn chế các bước có nguy cơ làm lệch màu sinh học của tổn thương:

- không dùng Gray-World mặc định
- CLAHE chỉ tác động nhẹ trên kênh L trong LAB
- giữ thông tin chroma tốt hơn
- vẫn hỗ trợ xóa lông để giảm nhiễu do hair artifact

Pipeline cũ vẫn được giữ dưới profile `legacy` để phục vụ tái lập hoặc ablation.

## Đánh giá

Các chỉ số cần báo cáo:

```text
AUC
Accuracy
F1-score
Precision
Recall / Sensitivity
Specificity
```

Ngoài đánh giá nội bộ, dự án còn hỗ trợ cross-dataset testing:

```text
Train HAM10000 -> Test BCN20000
Train BCN20000 -> Test HAM10000
```


