import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Thiết lập style cho matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Màu cho 4 mode
mode_labels = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']

def read_history_data(base_path, dataset_folder, dataset_code, model_name='effb4'):
    """
    Đọc dữ liệu history từ các mode
    dataset_folder: tên thư mục (ham10000, bcn20000)
    dataset_code: mã trong tên file (ham10k, bcn20k)
    """
    data = {}
    for mode_idx in range(1, 5):
        # Tên thư mục có thể là "mode 1" hoặc "mode1"
        mode_folder1 = f"mode {mode_idx}"
        mode_folder2 = f"mode{mode_idx}"
        
        # Tên file: history_ham10k_effb4.csv hoặc history_bcn20k_effb4.csv
        filename = f'history_{dataset_code}_{model_name}.csv'
        
        # Thử cả 2 format tên thư mục
        history_path1 = os.path.join(base_path, dataset_folder, mode_folder1, filename)
        history_path2 = os.path.join(base_path, dataset_folder, mode_folder2, filename)
        
        if os.path.exists(history_path1):
            history_path = history_path1
        elif os.path.exists(history_path2):
            history_path = history_path2
        else:
            print(f"Warning: Không tìm thấy file {history_path1} hoặc {history_path2}")
            continue
            
        try:
            df = pd.read_csv(history_path, encoding='utf-8-sig')
            # Đảm bảo cột epoch tồn tại (xử lý BOM nếu có)
            df.columns = df.columns.str.strip()
            data[mode_idx] = df
            print(f"Đã đọc: {history_path}")
        except Exception as e:
            print(f"Lỗi khi đọc {history_path}: {e}")
    
    return data

def read_test_metrics(base_path, dataset_folder, dataset_code, model_name='effb4'):
    """
    Đọc dữ liệu test metrics từ các mode
    dataset_folder: tên thư mục (ham10000, bcn20000)
    dataset_code: mã trong tên file (ham10k, bcn20k)
    """
    data = {}
    for mode_idx in range(1, 5):
        # Tên thư mục có thể là "mode 1" hoặc "mode1"
        mode_folder1 = f"mode {mode_idx}"
        mode_folder2 = f"mode{mode_idx}"
        
        # Tên file có thể có hoặc không có suffix _mode{idx}
        # Ví dụ: test_metrics_bcn20k_effb4.csv hoặc test_metrics_bcn20k_effb4_mode2.csv
        filename_no_suffix = f'test_metrics_{dataset_code}_{model_name}.csv'
        filename_with_suffix = f'test_metrics_{dataset_code}_{model_name}_mode{mode_idx}.csv'
        
        # Thử tất cả các kết hợp
        possible_paths = [
            os.path.join(base_path, dataset_folder, mode_folder1, filename_no_suffix),
            os.path.join(base_path, dataset_folder, mode_folder2, filename_no_suffix),
            os.path.join(base_path, dataset_folder, mode_folder1, filename_with_suffix),
            os.path.join(base_path, dataset_folder, mode_folder2, filename_with_suffix),
        ]
        
        test_path = None
        for path in possible_paths:
            if os.path.exists(path):
                test_path = path
                break
        
        if test_path is None:
            print(f"Warning: Không tìm thấy file test metrics cho mode {mode_idx}")
            continue
            
        try:
            df = pd.read_csv(test_path, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()
            data[mode_idx] = df
            print(f"Đã đọc: {test_path}")
        except Exception as e:
            print(f"Lỗi khi đọc {test_path}: {e}")
    
    return data

def plot_training_metrics(history_data, metric_name, dataset_name, model_name, output_dir):
    """
    Vẽ biểu đồ cho các chỉ số training/validation theo epoch
    """
    plt.figure(figsize=(12, 6))
    
    for mode_idx, df in history_data.items():
        if metric_name in df.columns:
            # Chỉ lấy dữ liệu đến epoch 11
            df_filtered = df[df['epoch'] <= 11]
            epochs = df_filtered['epoch'].values
            values = df_filtered[metric_name].values
            plt.plot(epochs, values, color=colors[mode_idx-1], 
                    label=mode_labels[mode_idx-1], linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    
    # Đặt tên trục y dựa trên metric
    if 'loss' in metric_name:
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        metric_title = 'Loss'
    elif 'acc' in metric_name:
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        metric_title = 'Accuracy'
    elif 'auc' in metric_name:
        plt.ylabel('AUC', fontsize=12, fontweight='bold')
        metric_title = 'AUC'
    elif 'f1' in metric_name:
        plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
        metric_title = 'F1-Score'
    else:
        plt.ylabel(metric_name, fontsize=12, fontweight='bold')
        metric_title = metric_name
    
    # Đặt tiêu đề
    if metric_name.startswith('train_'):
        phase = 'Training'
    elif metric_name.startswith('val_'):
        phase = 'Validation'
    else:
        phase = ''
    
    title = f'{phase} {metric_title} - {dataset_name} ({model_name.upper()})'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu file
    output_path = os.path.join(output_dir, f'{metric_name}_{dataset_name}_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {output_path}")

def plot_test_metrics(test_data, metric_name, dataset_name, model_name, output_dir):
    """
    Vẽ biểu đồ đường cho các chỉ số test (1 giá trị cho mỗi mode)
    """
    plt.figure(figsize=(10, 6))
    
    modes = []
    values = []
    plot_colors = []
    
    for mode_idx in range(1, 5):
        if mode_idx in test_data:
            df = test_data[mode_idx]
            if metric_name in df.columns:
                modes.append(mode_idx)
                values.append(df[metric_name].values[0])
                plot_colors.append(colors[mode_idx-1])
    
    # Vẽ line chart với marker
    plt.plot(modes, values, color='#333333', linewidth=2, linestyle='--', alpha=0.5)
    
    # Vẽ các điểm với màu khác nhau cho mỗi mode
    for i, (mode, value, color) in enumerate(zip(modes, values, plot_colors)):
        plt.scatter(mode, value, color=color, s=150, zorder=5, label=mode_labels[mode-1], edgecolors='black', linewidths=1.5)
        plt.annotate(f'{value:.4f}', (mode, value), textcoords="offset points", 
                    xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Mode', fontsize=12, fontweight='bold')
    plt.xticks(modes, [f'Mode {m}' for m in modes], fontsize=11)
    
    # Đặt tên trục y dựa trên metric
    if metric_name == 'loss':
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        metric_title = 'Loss'
    elif metric_name == 'acc':
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        metric_title = 'Accuracy'
    elif metric_name == 'auc':
        plt.ylabel('AUC', fontsize=12, fontweight='bold')
        metric_title = 'AUC'
    elif metric_name == 'f1':
        plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
        metric_title = 'F1-Score'
    else:
        plt.ylabel(metric_name, fontsize=12, fontweight='bold')
        metric_title = metric_name
    
    title = f'Test {metric_title} - {dataset_name} ({model_name.upper()})'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Lưu file
    output_path = os.path.join(output_dir, f'test_{metric_name}_{dataset_name}_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {output_path}")

def plot_all_metrics_for_model(base_path, model_folder, model_name, output_base_dir):
    """
    Vẽ tất cả các biểu đồ cho một model
    """
    model_path = os.path.join(base_path, model_folder)
    
    # Danh sách các dataset: folder_name -> (file_code, model_code)
    # model_code sử dụng model_name được truyền vào
    datasets = {
        'ham10000': ('ham10k', model_name),
        'bcn20000': ('bcn20k', model_name)
    }
    
    # Các chỉ số cần vẽ
    train_metrics = ['train_acc', 'train_loss', 'train_auc', 'train_f1']
    val_metrics = ['val_acc', 'val_loss', 'val_auc', 'val_f1']
    
    for dataset_folder, (dataset_code, model_code) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Đang xử lý: {model_name.upper()} - {dataset_folder.upper()}")
        print(f"{'='*60}")
        
        # Tạo thư mục output
        output_dir = os.path.join(output_base_dir, model_folder, dataset_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # Đọc dữ liệu history
        history_data = read_history_data(model_path, dataset_folder, dataset_code, model_code)
        
        # Vẽ các chỉ số training
        print(f"\nVẽ biểu đồ Training metrics...")
        for metric in train_metrics:
            try:
                plot_training_metrics(history_data, metric, dataset_folder, model_name, output_dir)
            except Exception as e:
                print(f"Lỗi khi vẽ {metric}: {e}")
        
        # Vẽ các chỉ số validation
        print(f"\nVẽ biểu đồ Validation metrics...")
        for metric in val_metrics:
            try:
                plot_training_metrics(history_data, metric, dataset_folder, model_name, output_dir)
            except Exception as e:
                print(f"Lỗi khi vẽ {metric}: {e}")

def main():
    # Đường dẫn gốc
    base_path = r'e:\NCKH\skin_cancer_project\src\evaluate'
    output_base_dir = r'e:\NCKH\skin_cancer_project\src\evaluate\plots'
    
    # Danh sách các model cần vẽ
    models = {
        # 'checkpoint_efcnb4': 'effb4',  
        'checkpoint_conv': 'conv',

    }
    
    print("="*60)
    print("BẮT ĐẦU VẼ BIỂU ĐỒ CHO TẤT CẢ CÁC MODEL")
    print("="*60)
    
    for model_folder, model_name in models.items():
        print(f"\n{'#'*60}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'#'*60}")
        
        try:
            plot_all_metrics_for_model(base_path, model_folder, model_name, output_base_dir)
        except Exception as e:
            print(f"Lỗi khi xử lý model {model_name}: {e}")
    
    print("\n" + "="*60)
    print("HOÀN THÀNH!")
    print("="*60)
    print(f"\nTất cả các biểu đồ đã được lưu vào: {output_base_dir}")

if __name__ == "__main__":
    main()
