# COCO_Classification_Project

MS COCO 物體分類與 ONNX 部署

## 專案描述
本專案針對 MS COCO 2017 資料集選定 4 種類別物件進行圖像分類，核心流程包含：
- 裁剪物件 ROI 作為訓練樣本
- 使用 ResNet50 模型訓練分類器
- 確保單類別準確率 > 95%
- 模型轉換為 ONNX 格式並部署測試

選定類別：
1. baseball bat
2. frisbee
3. laptop
4. stop sign

---

## 環境需求
- Python 3.x
- TensorFlow 2.x
- ONNX / tf2onnx
- onnxruntime
- numpy, pandas, matplotlib, Pillow

---

## 流程與指令

### 第一步：檢查數據量 (可選)
```bash
python check_amount.py
```
功能：統計 COCO 數據集各類別實例數量。

### 第二步：數據準備 (ROI 裁剪)
```bash
python crop.py
```
功能：裁剪指定類別物件 ROI，生成訓練/測試圖片。

### 第三步：模型訓練
```bash
python train.py --config config.ini
```
功能：載入 ResNet50 並訓練模型。

### 第四步：TensorFlow 模型評估
```bash
python test.py
```
功能：生成分類報告與混淆矩陣。

### 第五步：轉換為 ONNX
```bash
python to_onnx.py
```
功能：轉換訓練好的 Keras 模型為 ONNX。

### 第六步：ONNX 模型測試
```bash
python onnx_test.py
```
功能：載入 ONNX 模型進行推論與評估。

---

## 輸出範例
- 每類別正確率、總體正確率
- 混淆矩陣圖
- 錯誤分類檔案列表 (例如 Frisbee 錯誤樣本)

---

## 作者
Sean。
