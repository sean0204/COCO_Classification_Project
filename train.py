import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# === 固定參數 ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-5
FREEZE_LAYERS = 2
PRETRAIN_WEIGHT_PATH = r'D:\Simon_Tool\Simon_Tool\Tools\Train_Predict_Tool\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
TRAIN_CSV = r'D:\test_simply\train.csv'
VAL_CSV = r'D:\test_simply\val.csv'
TEST_CSV = r'D:\test_simply\test.csv'
TRAIN_DIR = r'D:\test_simply\Train'
FINAL_MODEL_PATH = r'D:\test_simply\Logs\model_final_new.h5'
LOG_DIR = r'D:\test_simply\Logs'

# === 自動讀取 class names ===
CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
print(f"Detected CLASS_NAMES: {CLASS_NAMES}")

# === 資料增強 ===
def augmentations(img, label):
    img = tf.image.random_hue(img, 0.1)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_flip_left_right(img)
    return img, label

# === 建 dataset ===
def build_dataset(df, augment=False, repeat=False):
    file_paths = df['file_path'].values
    labels = df['label'].apply(lambda x: CLASS_NAMES.index(x)).values
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    def process(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augmentations, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.shuffle(1000).repeat()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# === 訓練流程 ===
def run_training():
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    train_ds = build_dataset(df_train, augment=True, repeat=True)
    val_ds = build_dataset(df_val)

    class_count = df_train['label'].value_counts().reindex(CLASS_NAMES).values
    min_count = np.min(class_count)
    class_weight = {i: min_count / c for i, c in enumerate(class_count)}
    print(f"Class weight: {class_weight}")

    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=IMAGE_SIZE + (3,))
    base_model.load_weights(PRETRAIN_WEIGHT_PATH)
    for layer in base_model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    os.makedirs(LOG_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, min_delta=0.001, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(LOG_DIR, "model_best.h5"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        tf.keras.callbacks.CSVLogger(os.path.join(LOG_DIR, "training.log"))
    ]

    steps_per_epoch = len(df_train) // BATCH_SIZE
    validation_steps = len(df_val) // BATCH_SIZE

    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks
    )

    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH}")

# === 測試流程 ===
def run_testing():
    df_test = pd.read_csv(TEST_CSV)
    test_ds = build_dataset(df_test)

    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # 畫混淆矩陣
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

# === 主流程 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['train', 'test'], required=True, help="模式選擇：train 或 test")
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "test":
        run_testing()
