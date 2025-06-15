
import onnxruntime as ort
import numpy as np
import os
from PIL import Image
import sys
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# --- 參數配置 ---
# 圖像尺寸，必須與模型訓練時的輸入尺寸一致
IMAGE_SIZE = (224, 224)
# 您分類的類別名稱列表，順序必須與模型輸出類別的索引順序一致
CLASS_NAMES = ['baseball_bat', 'frisbee', 'laptop', 'stop_sign']
# ONNX模型檔案的路徑
ONNX_MODEL_PATH = r'D:\test_simply\Logs\final_weight(baseball_bat)_120_0615.onnx' # 確保這裡的路徑正確

# 測試集圖像的根目錄，例如 D:\test_simply\Test\baseball_bat, D:\test_simply\Test\frisbee 等
TEST_DIR = r'D:\project_model\Test' # 確保這裡的路徑正確

# === 圖像預處理函數 ===
# 該函數負責將單張圖像載入、調整大小、歸一化，並轉換為ONNX模型所需的輸入格式。
def preprocess(img_path):
    try:
        img = Image.open(img_path).convert("RGB") # 載入圖像並確保是RGB格式
        img = img.resize(IMAGE_SIZE) # 調整圖像大小到模型期望的尺寸
        img_array = np.array(img).astype(np.float32) / 255.0 # 轉換為NumPy數組，數據類型為float32，並歸一化到0-1
        # ONNX模型通常期望 [Batch_size, Channels, Height, Width] 或 [Batch_size, Height, Width, Channels]
        # ResNet通常期望 [Batch_size, Height, Width, Channels] (NHWC) 如果是從TensorFlow轉換的
        # 因此，這裡移除了將 HWC 轉換為 CHW 的轉置操作。
        # 如果您的模型確實期望 NCHW 格式，請將下一行註釋掉的代碼取消註釋
        # img_array = np.transpose(img_array, (2, 0, 1)) # 將 HWC 轉換為 CHW
        img_array = np.expand_dims(img_array, axis=0) # 增加批次維度 [1, Height, Width, Channels] (NHWC)
        return img_array
    except Exception as e:
        print(f"錯誤：預處理圖像 '{img_path}' 失敗：{e}。跳過。", file=sys.stderr)
        return None

# === 載入ONNX模型 ===
# 使用 onnxruntime 載入並初始化ONNX模型會話。
try:
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    # 獲取模型輸入和輸出的名稱，這是ONNX推理時必需的
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX 模型 '{ONNX_MODEL_PATH}' 載入成功。")
    print(f"模型輸入名稱: {input_name}, 輸出名稱: {output_name}")
except Exception as e:
    print(f"錯誤：無法載入 ONNX 模型 '{ONNX_MODEL_PATH}'。請檢查路徑或模型文件。錯誤信息: {e}", file=sys.stderr)
    sys.exit(1)

# === 初始化用於評估的列表和字典 ===
# 儲存所有真實標籤和預測標籤，以便計算混淆矩陣
all_y_true = []
all_y_pred = []

# === 執行測試和推論 ===
print("\n正在使用 ONNX 模型對測試集進行推論...")
total_images_processed = 0
total_classes_found = 0

for class_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f"警告：測試類別資料夾 '{class_dir}' 不存在。跳過此類別。")
        continue
    
    total_images_in_class = len([name for name in os.listdir(class_dir) if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
    if total_images_in_class == 0:
        print(f"警告：類別資料夾 '{class_dir}' 中沒有圖像。跳過此類別。")
        continue

    total_classes_found += 1
    
    # 遍歷每個類別資料夾中的圖像
    for img_name in os.listdir(class_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(class_dir, img_name)
            img_array = preprocess(img_path)
            
            if img_array is None: # 如果預處理失敗，則跳過
                continue

            try:
                # 執行ONNX模型推論
                result = session.run([output_name], {input_name: img_array})
                # 獲取預測結果中機率最高的類別索引
                pred_idx = np.argmax(result[0], axis=1)[0]

                all_y_true.append(class_idx) # 真實標籤是當前類別的索引
                all_y_pred.append(pred_idx) # 預測標籤

                total_images_processed += 1
            except Exception as e:
                print(f"\n錯誤：ONNX 推論圖像 '{img_path}' 失敗：{e}。跳過。", file=sys.stderr)
                continue

print(f"\n推論完成。共處理 {total_images_processed} 張圖像。")
if total_images_processed == 0:
    print("沒有任何圖像被成功處理。請檢查您的圖像路徑和ONNX模型。")
    sys.exit(1)


# === 輸出結果 ===
# 使用 sklearn 打印分類報告
print("\n==== ONNX 模型分類報告 ====")
# 確保 y_true 和 y_pred 有數據
if len(all_y_true) > 0:
    print(classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, zero_division=0))
else:
    print("沒有足夠的數據來生成分類報告。")

# 計算並打印混淆矩陣
cm = confusion_matrix(all_y_true, all_y_pred)
print("\n==== ONNX 模型混淆矩陣 ====")
print(cm)

# 計算並打印每類別準確率
print("\n==== ONNX 模型每類別準確率 (基於混淆矩陣 TP, TN, FP, FN) ====")
for i, class_name in enumerate(CLASS_NAMES):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    tn = np.sum(cm) - tp - fp - fn

    total_relevant = tp + tn + fp + fn
    acc = (tp + tn) / total_relevant if total_relevant > 0 else 0
    
    print(f"{class_name}: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Accuracy = {acc:.4f}")

# 計算並打印總體準確率
overall_acc = np.mean(np.array(all_y_true) == np.array(all_y_pred))
print(f"\n==== ONNX 模型總體準確率 ====")
print(f"總體正確率: {overall_acc:.4f}")

# 繪製混淆矩陣圖
plt.figure(figsize=(8, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('ONNX Model Confusion Matrix', fontsize=16)
plt.colorbar()
tick_marks = np.arange(len(CLASS_NAMES))
plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha='right', fontsize=12)
plt.yticks(tick_marks, CLASS_NAMES, fontsize=12)

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=10)
plt.tight_layout()
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()
