import numpy as np
import time
import tflite_runtime.interpreter as tflite
from PIL import Image
# 加载标签文件
with open('labels.txt', 'r') as f:
    labels = f.read().splitlines()

# 加载图像
image_path = 'grace_hopper.bmp'  # 替换为您的图像路径
image = Image.open(image_path)
image = image.resize((224, 224))
# 归一化
image = np.array(image,dtype="float32") / 255.0

# 加载TFLite模型
interpreter = tflite.Interpreter(model_path="mobilenet_v1_1.0_224.tflite")
interpreter.allocate_tensors()
start_time = time.perf_counter()
# 获取输入和输出张量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 输入图像并进行推理
input_shape = input_details[0]['shape']
input_data = np.expand_dims(image, axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
end_time = time.perf_counter()
execution_time = end_time - start_time
print("执行时间：", execution_time, "秒")
# 获取输出
output_data = interpreter.get_tensor(output_details[0]['index'])

# 打印预测结果
predicted_label_index = np.argmax(output_data)
predicted_label = labels[predicted_label_index]
confidence = output_data[0][predicted_label_index]

print("Predicted label:", predicted_label)
print("Confidence:", confidence)
