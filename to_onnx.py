import tf2onnx
import tensorflow as tf

# 載入 Keras H5 模型
model = tf.keras.models.load_model(r"D:\project_model\final_weight(baseball_bat)_150sucess.h5")

# 轉成 ONNX graph
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = r"D:\project_model\final_weight(baseball_bat)_150sucess.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

print(f"ONNX model saved to {output_path}")
