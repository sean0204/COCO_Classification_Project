import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# === 參數 ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
CLASS_NAMES = ['baseball_bat', 'frisbee', 'laptop', 'stop_sign']
MODEL_PATH = r'D:\test_simply\Logs\final_weight(baseball_bat)_120_0615.h5'
TEST_DIR = r'D:\test_simply\Test'
TEST_CSV = r'D:\test_simply\test.csv'

# === 自動產生 test.csv ===
def build_test_csv(test_dir, csv_path, class_names):
    rows = []
    for class_name in class_names:
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                rows.append({
                    'file_path': os.path.join(class_path, img_name),
                    'label': class_name
                })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"產生 test.csv 完成，共 {len(df)} 筆資料：{csv_path}")
    return df

df_test = build_test_csv(TEST_DIR, TEST_CSV, CLASS_NAMES)

# === 建 dataset ===
def build_dataset(df, class_names):
    file_paths = df['file_path'].values
    labels = df['label'].apply(lambda x: class_names.index(x)).values
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    return ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = build_dataset(df_test, CLASS_NAMES)

# === 載模型 + 預測 ===
model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# === 輸出結果 ===
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# === 混淆矩陣 ===
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# === 每類正確數、總數、正確率 ===
print("\nPer-class accuracy:")
for i, class_name in enumerate(CLASS_NAMES):
    total = np.sum(np.array(y_true) == i)
    correct = cm[i, i]
    acc = correct / total if total > 0 else 0
    print(f"{class_name}: {correct}/{total} 正確率 = {acc:.4f}")

# === 總體正確率 ===
overall_acc = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\n總體正確率: {overall_acc:.4f}")

# === 畫混淆矩陣 ===
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

