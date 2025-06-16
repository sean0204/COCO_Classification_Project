#!/usr/bin/env python3
# coding: utf-8
"""
單一腳本包含資料切分、訓練和評估，使用 TensorFlow 2.x 的 ResNet50 模型。
用法：
    python combined_script.py --config path/to/config.ini [--data_dir path/to/dataset]

資料夾結構應包含子資料夾 `Train/` 和 `Test/`，每個底下依分類放影像。
腳本會依照 config 產生 `train.csv`, `valid.csv`, `test.csv`，
自動切分資料、訓練並只儲存最終模型。
"""
import os
import argparse
import logging
import configparser
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# 設定 TF v1 GPU 記憶體比例，避免一次佔滿 GPU
config_proto = ConfigProto()
config_proto.gpu_options.per_process_gpu_memory_fraction = 0.5
session = InteractiveSession(config=config_proto)
AUTOTUNE = tf.data.experimental.AUTOTUNE

# ----------------------------------------------------------------------------
# Config 類別：讀取並管理 config.ini 設定
# ----------------------------------------------------------------------------
class Config:
    def __init__(self, cfg_path):
        parser = configparser.ConfigParser()
        parser.read(cfg_path)
        sec = parser['SETTING']
        # 讀取基本訓練參數
        self.BATCH_SIZE        = sec.getint('batch_size', 8)
        self.epoch_num         = sec.getint('epoch', 100)
        self.IMAGE_SIZE        = tuple(int(x) for x in sec.get('image_size','(224,224)').strip('()').split(','))
        self.learning_rate     = sec.getfloat('learningrate', 1e-4)
        self.loss_func         = sec.get('loss_func','categorical_crossentropy')
        self.loss_metrics      = sec.get('loss_metrics','accuracy')
        self.freeze_layer      = sec.getint('freeze_layer', 2)
        # 讀取檔案路徑設定
        base = Path(cfg_path).parent
        self.trainCsvPath      = Path(sec.get('traincsvpath', str(base/'train.csv')))
        self.validCsvPath      = Path(sec.get('validcsvpath', str(base/'valid.csv')))
        self.testCsvPath       = Path(sec.get('testcsvpath',  str(base/'test.csv')))
        self.LOG_PATH          = Path(sec.get('logpath', str(base/'Logs')))
        self.WEIGHTS_FINAL     = Path(sec.get('finalweight', str(base/'final_weight.h5')))
        self.pre_train_weight  = Path(sec.get('pretrain_weight', 'imagenet'))
        # EarlyStopping 相關設定
        if 'EARLY_STOP' in parser:
            esp = parser['EARLY_STOP']
            self.IsEarlyStop    = True
            self.EARLY_PATIENCE = esp.getint('early_patience', 50)
            self.EARLY_MIN_DELTA= esp.getfloat('early_min_delta', 0.0)
            self.EARLY_MONITOR  = esp.get('early_monitor','val_loss')
            self.EARLY_MODE     = esp.get('early_mode','min')
        else:
            self.IsEarlyStop    = False

    def display(self):
        """列印當前設定值，方便除錯"""
        print("\n----- 配置 -----")
        for k, v in self.__dict__.items():
            print(f"{k:18s} = {v}")
        print("--------------\n")

# ----------------------------------------------------------------------------
# 資料集檔案處理函式
# ----------------------------------------------------------------------------
def get_img_label(root):
    """
    掃描資料夾下所有分類子資料夾名稱
    :param root: 根路徑 (Train/ 或 Test/)
    :return: 分類標籤列表
    """
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


def get_img_path(label, root):
    """
    取得指定分類資料夾下所有影像檔路徑
    :param label: 分類名稱
    :param root: 根路徑
    :return: 影像路徑清單
    """
    out = []
    for r, _, fs in os.walk(os.path.join(root, label)):
        for f in fs:
            if f.lower().endswith(('png','jpg','jpeg')):
                out.append(os.path.join(r, f))
    return out


def dict_to_df(data):
    """
    將 {label: [paths]} 轉成 DataFrame，包含 filename, label 欄位
    :param data: 字典
    :return: pandas.DataFrame
    """
    return pd.DataFrame([{'filename': p, 'label': l} for l, ps in data.items() for p in ps])

# ----------------------------------------------------------------------------
# 資料切分函式
# ----------------------------------------------------------------------------
def df_split(df, tr, vl, shuffle=True):
    """
    按比例將 DataFrame 切成 train, valid, test
    :param df: 原始 DataFrame
    :param tr: 訓練集比例
    :param vl: 驗證集比例
    :param shuffle: 是否打散順序
    :return: df_train, df_valid, df_test
    """
    df_tr = pd.DataFrame(columns=df.columns)
    df_vl = pd.DataFrame(columns=df.columns)
    df_te = pd.DataFrame(columns=df.columns)
    for lbl in df['label'].unique():
        sub = df[df['label']==lbl]
        if shuffle: sub = sub.sample(frac=1)
        n = len(sub)
        i1 = int(n * tr)
        i2 = int(n * (tr + vl))
        df_tr = pd.concat([df_tr, sub.iloc[:i1]])
        df_vl = pd.concat([df_vl, sub.iloc[i1:i2]])
        df_te = pd.concat([df_te, sub.iloc[i2:]])
    if shuffle:
        return df_tr.sample(frac=1), df_vl.sample(frac=1), df_te.sample(frac=1)
    return df_tr, df_vl, df_te

# ----------------------------------------------------------------------------
# 圖片處理及資料增強函式
# ----------------------------------------------------------------------------
def decode_img(img_bytes, w, h):
    """
    解碼 JPEG bytes -> float32 tensor 並調整尺寸
    :param img_bytes: 圖片二進位數據
    :param w: 寬度
    :param h: 高度
    :return: TensorFlow tensor
    """
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [w, h])


def augment(img, lbl):
    """
    隨機色相、旋轉與左右翻轉增強
    :param img: 影像 tensor
    :param lbl: 標籤 tensor
    """
    img = tf.image.random_hue(img, 0.1)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, tf.int32))
    img = tf.image.random_flip_left_right(img)
    return img, lbl

# ----------------------------------------------------------------------------
# 建構 tf.data 資料管線函式
# ----------------------------------------------------------------------------
def make_dataset(df, cfg, mode='train'):
    """
    由 DataFrame 建立 tf.data.Dataset，train 模式下套用增強
    :param df: 需含 'filename', 'label_idx'
    :param cfg: Config 物件
    :param mode: 'train' 或 'valid'
    """
    paths = df['filename'].values
    labels= df['label_idx'].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda f,l: (decode_img(tf.io.read_file(f), *cfg.IMAGE_SIZE), tf.one_hot(l, cfg.num_classes)), AUTOTUNE)
    if mode=='train': ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1000).repeat().batch(cfg.BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ----------------------------------------------------------------------------
# 計算類別權重函式
# ----------------------------------------------------------------------------
def calc_class_weight(df, num_classes):
    """
    計算每類別權重，用於 class_weight
    :param df: 含 'label_idx' 欄位的 DataFrame
    :param num_classes: 類別總數
    """
    counts = df['label_idx'].value_counts().reindex(range(num_classes), fill_value=0).values
    nonzero = counts[counts>0]
    if len(nonzero)==0:
        return {i:1.0 for i in range(num_classes)}
    m = nonzero.min()
    return {i:(m/counts[i] if counts[i]>0 else 0.0) for i in range(num_classes)}

# ----------------------------------------------------------------------------
# 訓練流程函式
# ----------------------------------------------------------------------------
def train(cfg):
    """
    執行從讀取 CSV -> 前處理 -> 建模 -> 訓練 -> 儲存模型的完整流程
    """
    # 讀取並過濾 CSV
    df_tr = pd.read_csv(cfg.trainCsvPath)
    df_vl = pd.read_csv(cfg.validCsvPath)
    df_tr = df_tr[df_tr['filename'].apply(os.path.isfile)].reset_index(drop=True)
    df_vl = df_vl[df_vl['filename'].apply(os.path.isfile)].reset_index(drop=True)
    # 標籤映射
    classes = sorted(df_tr['label'].unique())
    cfg.num_classes = len(classes)
    df_tr['label_idx'] = df_tr['label'].map({c:i for i,c in enumerate(classes)})
    df_vl['label_idx'] = df_vl['label'].map({c:i for i,c in enumerate(classes)})
    # 計算權重 & 建立 Dataset
    cw = calc_class_weight(df_tr, cfg.num_classes)
    ds_tr = make_dataset(df_tr, cfg, mode='train')
    ds_vl = make_dataset(df_vl, cfg, mode='valid')
    # 回呼設定
    cbks = [keras.callbacks.CSVLogger(str(cfg.LOG_PATH/'training.log'))]
    if cfg.IsEarlyStop:
        cbks.append(keras.callbacks.EarlyStopping(monitor=cfg.EARLY_MONITOR,
                                                 patience=cfg.EARLY_PATIENCE,
                                                 min_delta=cfg.EARLY_MIN_DELTA,
                                                 mode=cfg.EARLY_MODE))
    # 架構 & 編譯模型
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    with strategy.scope():
        base = keras.applications.ResNet50(include_top=False, weights=str(cfg.pre_train_weight),
                                           input_shape=(*cfg.IMAGE_SIZE,3))
        x = keras.layers.GlobalAveragePooling2D()(base.output)
        out = keras.layers.Dense(cfg.num_classes, activation='softmax')(x)
        model = keras.Model(base.input, out)
        for layer in model.layers[:cfg.freeze_layer]: layer.trainable=False
        for layer in model.layers[cfg.freeze_layer:]: layer.trainable=True
        model.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate),
                      loss=cfg.loss_func, metrics=[cfg.loss_metrics])
    # 執行訓練並儲存最終模型
    steps_tr = len(df_tr)//cfg.BATCH_SIZE
    steps_vl = len(df_vl)//cfg.BATCH_SIZE
    cfg.LOG_PATH.mkdir(parents=True, exist_ok=True)
    model.fit(ds_tr, epochs=cfg.epoch_num, steps_per_epoch=steps_tr,
              validation_data=ds_vl, validation_steps=steps_vl,
              class_weight=cw, callbacks=cbks)
    model.save(str(cfg.WEIGHTS_FINAL))
    tf.keras.backend.clear_session()

# ----------------------------------------------------------------------------
# 推論流程函式
# ----------------------------------------------------------------------------
def predict(cfg):
    """
    執行從讀取 CSV -> 前處理 -> 模型載入 -> 預測 -> 儲存結果的完整流程
    """
    df_te = pd.read_csv(cfg.testCsvPath)
    df_te = df_te[df_te['filename'].apply(os.path.isfile)].reset_index(drop=True)
    classes = sorted(pd.read_csv(cfg.trainCsvPath)['label'].unique())
    cfg.num_classes = len(classes)
    df_te['label_idx'] = df_te['label'].map({c:i for i,c in enumerate(classes)})
    ds_te = tf.data.Dataset.from_tensor_slices((df_te['filename'].values, df_te['label_idx'].values))
    ds_te = ds_te.map(lambda f,l: (decode_img(tf.io.read_file(f), *cfg.IMAGE_SIZE), tf.one_hot(l, cfg.num_classes)), AUTOTUNE).batch(cfg.BATCH_SIZE)
    model = keras.models.load_model(str(cfg.WEIGHTS_FINAL))
    preds = model.predict(ds_te)
    rows = []
    for (fn, lbl), p in zip(zip(df_te['filename'], df_te['label']), preds):
        idx = np.argmax(p)
        rows.append({'filename':fn, 'DatasetLabel':lbl, 'PredictLabel':classes[idx], 'PredictValue':float(p[idx])})
    pd.DataFrame(rows).to_csv(str(cfg.LOG_PATH/'test_result.csv'), index=False)
    tf.keras.backend.clear_session()

# ----------------------------------------------------------------------------
# 主程式
# ----------------------------------------------------------------------------
def main():
    """主程式：讀取設定、切分資料、執行訓練與推論"""
    parser = argparse.ArgumentParser(description='Pipeline: split, train, predict')
    parser.add_argument('-c','--config', required=True, help='config.ini 路徑')
    parser.add_argument('-d','--data_dir', default=None, help='包含 Train/ 和 Test/ 的根目錄')
    args = parser.parse_args()
    cfg = Config(args.config)
    cfg.display()
    # 設定日誌
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.INFO)
    cfg.LOG_PATH.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(cfg.LOG_PATH/'pipeline.log'))
    fh.setFormatter=logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    logger.addHandler(fh)
    # 確定資料根目錄
    root = Path(args.data_dir) if args.data_dir else Path(args.config).parent
    # 切分 Train/Valid
    train_root = root/'Train'
    if train_root.is_dir():
        data = {lbl:get_img_path(lbl,str(train_root)) for lbl in get_img_label(str(train_root))}
        df_tr, df_vl, _ = df_split(dict_to_df(data), 0.8, 0.2)
        df_tr.to_csv(str(cfg.trainCsvPath), index=False)
        df_vl.to_csv(str(cfg.validCsvPath), index=False)
        logger.info(f'寫入 train.csv ({len(df_tr)}) & valid.csv ({len(df_vl)})')
    # 切分 Test
    test_root = root/'Test'
    if test_root.is_dir():
        data = {lbl:get_img_path(lbl,str(test_root)) for lbl in get_img_label(str(test_root))}
        dict_to_df(data).to_csv(str(cfg.testCsvPath), index=False)
        logger.info('寫入 test.csv')
    # 執行訓練與推論
    train(cfg)
    logger.info('訓練完成')
    predict(cfg)
    logger.info('推論完成')

if __name__=='__main__':
    main()
