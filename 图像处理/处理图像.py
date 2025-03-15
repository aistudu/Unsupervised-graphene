from skimage import exposure, filters,morphology
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import matplotlib
import os
import cv2
from skimage import io, color
from skimage import exposure, filters, feature
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
import cv2

output_directory = r'C:\Users\Administrator\Desktop\myai\unsupervised\pytorch-unsupervised-segmentation-tip-master\link\12131'
matplotlib.use('TkAgg')  # 使用TkAgg后端，这个后端适用于生成图像文件但不显示它们

# 读取图片
plt.rcParams['font.sans-serif'] = ['SimHei']
image_r = '6_acetic.jpeg'
image = io.imread(image_r)

output_r_123 = os.path.join(output_directory, '1111111.jpeg')
img_lab = color.rgb2lab(image)

img_lab_normalized = cv2.normalize(img_lab, None, 0, 255, cv2.NORM_MINMAX)
img_lab_normalized = np.uint8(img_lab_normalized)
cv2.imwrite(output_r_123, img_lab_normalized)

# 分别分析L/a/b通道的对比度
# 输出路径
output_r_zhifang = os.path.join(output_directory, 'zhifang.jpeg')
output_r_lvbo = os.path.join(output_directory, 'lvbo.jpeg')
output_r_bianyuan = os.path.join(output_directory, 'bianyuan.jpeg')
output_r_yuzhi = os.path.join(output_directory, 'yuzhi.jpeg')

# 提取RGB通道
image_r = image[:, :, 0]
# 对比度增强（拉伸直方图）
image_enhanced = exposure.equalize_hist(image_r )
# 高斯模糊去噪
image_smoothed = filters.gaussian(image_enhanced , sigma=1)



# 边缘检测：使用Canny算子进行边缘检测
edges = feature.canny(image_smoothed)
edges_uint8 = np.uint8(edges * 255)

# 阈值分割：使用Otsu方法自动计算阈值并进行分割
thresh_value = filters.threshold_otsu(image_smoothed)
binary_image = image_smoothed > thresh_value
binary_image_uint8 = np.uint8(binary_image * 255)  # 将布尔数组转换为0和255之间的uint8类型

# 将图像转换为 [0, 255] 范围内的 uint8 类型
image_enhanced_uint8 = np.uint8(np.clip(image_enhanced * 255, 0, 255))

# 保存图像
try:
    cv2.imwrite(output_r_zhifang, image_enhanced_uint8)  # 保存图像
    cv2.imwrite(output_r_lvbo, np.uint8(image_smoothed * 255))  # 保存滤波后的图像
    cv2.imwrite(output_r_bianyuan, edges_uint8)  # 保存边缘检测结果
    cv2.imwrite(output_r_yuzhi, binary_image_uint8)  # 保存二值化分割结果
    print("图像保存成功！")
except Exception as e:
    print(f"图像保存失败: {e}")

# 可视化增强后的图片、边缘检测结果和二值化结果
plt.figure(figsize=(10, 10))

# 显示滤波后的图片
plt.subplot(2, 3, 1)
plt.imshow(image_smoothed, cmap='gray')
plt.title('滤波后的图片')

# 显示增强后的图片
plt.subplot(2, 3, 2)
plt.imshow(image_enhanced, cmap='gray')
plt.title('增强后的图片')

# 显示边缘检测结果
plt.subplot(2, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('边缘检测结果')

# 显示二值化分割结果
plt.subplot(2, 3, 4)
plt.imshow(binary_image, cmap='gray')
plt.title('阈值分割结果')

plt.show()
