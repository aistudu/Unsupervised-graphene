import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import background_process
import sys
import cv2
import numpy as np
import cupy as cp
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, QWidget, QLineEdit, QLabel
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from skimage import io
from skimage import exposure, filters
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端，这个后端适用于生成图像文件但不显示它们
import matplotlib.pyplot as plt
import sklearn
plt.rcParams['font.sans-serif'] = ['SimHei']
use_cuda = torch.cuda.is_available()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('图像处理')
        self.setGeometry(100, 100, 400, 400)
        self.weight_cache=''
        self.input_image_path = ''
        self.inputpath_cache=''
        self.output_directory = ''
        self.crop_box = []
        self.last_cropped_image_path = ''
        layout = QVBoxLayout()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)




        # 参数设置
        self.nChannel_edit = QLineEdit(self)
        self.nChannel_edit.setPlaceholderText('nChannel (默认20)')
        self.nChannel_edit.setText('20')  # 默认值
        layout.addWidget(QLabel('nChannel:'))
        layout.addWidget(self.nChannel_edit)

        self.nConv_edit = QLineEdit(self)
        self.nConv_edit.setPlaceholderText('nConv (默认2)')
        self.nConv_edit.setText('2')  # 默认值
        layout.addWidget(QLabel('nConv:'))
        layout.addWidget(self.nConv_edit)

        self.maxIter_edit = QLineEdit(self)
        self.maxIter_edit.setPlaceholderText('maxIter (默认100)')
        self.maxIter_edit.setText('100')  # 默认值
        layout.addWidget(QLabel('maxIter:'))
        layout.addWidget(self.maxIter_edit)

        self.minLabels_edit = QLineEdit(self)
        self.minLabels_edit.setPlaceholderText('minLabels (默认4)')
        self.minLabels_edit.setText('4')  # 默认值
        layout.addWidget(QLabel('minLabels:'))
        layout.addWidget(self.minLabels_edit)

        self.lr_edit = QLineEdit(self)
        self.lr_edit.setPlaceholderText('lr (默认0.5)')
        self.lr_edit.setText('0.5')  # 默认值
        layout.addWidget(QLabel('学习率 (lr):'))
        layout.addWidget(self.lr_edit)
        # 11111111
        self.calibrate_button = QPushButton('校准图片')
        self.calibrate_button.clicked.connect(self.calibrate_image)
        layout.addWidget(self.calibrate_button)


        self.select_image_button = QPushButton('选择输入图像')
        self.select_image_button.clicked.connect(self.select_image)
        layout.addWidget(self.select_image_button)

        self.select_directory_button = QPushButton('选择输出文件夹')
        self.select_directory_button.clicked.connect(self.select_directory)
        layout.addWidget(self.select_directory_button)
        #权重选择

        self.weight_file_path = ''
        self.select_weight_button = QPushButton('选择权重文件')
        self.select_weight_button.clicked.connect(self.select_weight_file)
        layout.addWidget(self.select_weight_button)


        self.process_button = QPushButton('处理图像')
        self.process_button.clicked.connect(self.process_image)
        layout.addWidget(self.process_button)


    def select_weight_file(self):
        self.weight_file_path, _ = QFileDialog.getOpenFileName(self, '选择权重文件', '', 'PyTorch Model Files (*.pth)')
        if not self.weight_file_path:
            print("未选择任权重文件。")
            print(self.weight_file_path)
        else:

            print(f"选择的权重文件: {self.weight_file_path}")

    def select_image(self):
        self.inputpath_cache=''
        self.input_image_path, _ = QFileDialog.getOpenFileName(self, '选择输入图像文件')

        if not self.input_image_path:
            print("未选择图像文件。")
        else:
            print(f"选择的图像文件: {self.input_image_path}")

    def select_directory(self):
        self.output_directory = QFileDialog.getExistingDirectory(self, '选择输出文件夹')
        if not self.output_directory:
            print("未选择输出文件夹。")
        else:
            print(f"选择的输出文件夹: {self.output_directory}")

    def cropper(self, arginput):
        def onselect(eclick, erelease):
            self.crop_box = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)]


        # 打开图片
        image = Image.open(arginput)
        img_array = image.convert('RGB')

        fig, ax = plt.subplots()
        ax.imshow(img_array)

        # 创建 RectangleSelector
        toggle_selector = RectangleSelector(ax, onselect)
        plt.show()


        if self.crop_box:
            # 裁剪图片
            self.crop_box = [max(0, x) for x in self.crop_box]  # 确保裁剪区域不出界
            cropped_image = image.crop(self.crop_box)
            # 保存裁剪后的图片
            base, ext = os.path.splitext(os.path.basename(arginput))
            output_cropped_image_path = os.path.join(os.path.dirname(arginput), f'{base}_cropped{ext}')

            # 确保新文件不会被覆盖
            if os.path.exists(output_cropped_image_path):
                counter = 1
                while os.path.exists(os.path.join(os.path.dirname(arginput), f'{base}_cropped_{counter}{ext}')):
                    counter += 1
                output_cropped_image_path = os.path.join(os.path.dirname(arginput), f'{base}_cropped_{counter}{ext}')

            cropped_image.save(output_cropped_image_path)
            print(f"Cropped image saved as {output_cropped_image_path}")
            self.crop_box=[]
            return output_cropped_image_path
        else:
            print("No cropping area selected.")
            return arginput

    def calibrate_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, '选择输入图像文件')
        if not image_path:
            print("未选择任何图像文件。")
            return None  # 或者返回一个适当的默认值
        print(f"选择的未校准图像文件: {image_path}")
        image_pil = Image.open(image_path).convert("RGB")
        image_rgb = cp.array(image_pil)

        # 处理图像并获取基线区域
        base_regions = background_process.process_image(image_rgb)

        # 计算每个基本区域的平均颜色
        average_colors = []
        for x_center, y_center, radius in base_regions:
            points = background_process.get_circle_points((x_center, y_center), radius, image_rgb)
            average_color = cp.mean(points[:, 2:], axis=0)
            average_colors.append(average_color)

        # 将 average_colors 转换为 NumPy 数组，方便后续操作
        average_colors = cp.array(average_colors)

        # 计算所有平均值的均值
        mean_color = cp.mean(average_colors, axis=0)
        threshold = 20  # 设定阈值

        # 筛选有效的圆
        valid_base_regions = []
        valid_average_colors = []
        for i, (x_center, y_center, radius) in enumerate(base_regions):
            if cp.linalg.norm(average_colors[i] - mean_color) < threshold:
                valid_base_regions.append((x_center, y_center, radius))
                valid_average_colors.append(average_colors[i])

        # 更新 base_regions 和 average_colors
        base_regions = valid_base_regions
        average_colors = valid_average_colors

        # 打印结果
        print("有效的圆圈的中心点坐标与圆圈的半径:", base_regions)
        print("有效的圆圈的RGB三通道平均值:", average_colors)
        # 显示结果
        print("圆圈的中心点坐标与圆圈的半径:", base_regions)
        print("圆圈的RGB三通道平均值:", average_colors)

        image_rgb_cpu = cp.asnumpy(image_rgb)
        base_regions_cpu = cp.asnumpy(base_regions)

        # 显示标记了基本区域的图像
        for x_center, y_center, radius in base_regions:
            cv2.circle(image_rgb_cpu, (x_center, y_center), radius, (255, 0, 0), 2)
        plt.imshow(image_rgb_cpu)
        plt.savefig('image_circle.jpeg', dpi=600, format='jpg')
        # plt.show()

        # 计算平均值
        result = background_process.robust_mean(average_colors)
        print("平均出的背景RGB值：", result)

        # 分离通道
        r, g, b = image_pil.split()
        # 将R通道作为灰度图像
        gray_image_r = r.convert("L")
        gray_image_g = g.convert("L")
        gray_image_b = b.convert("L")
        # 初始化列表来保存每个通道的坐标和像素值
        coordinate_R = []
        coordinate_G = []
        coordinate_B = []

        # 直接从给定的列表中获取圆圈的中心点坐标和对应的 RGB 三通道平均值
        average_color = []
        for x, y, radius in base_regions:
            # 获取 RGB 三通道平均值
            average_color = average_colors[base_regions.index((x, y, radius))]
            coordinate_R.append([x, y, round(average_color[0].item())])
            coordinate_G.append([x, y, round(average_color[1].item())])
            coordinate_B.append([x, y, round(average_color[2].item())])

        # 打印结果
        print("R通道坐标和像素值:", coordinate_R)
        print("G通道坐标和像素值:", coordinate_G)
        print("B通道坐标和像素值:", coordinate_B)

        # 将列表转换为 NumPy 数组
        points_R = cp.array(coordinate_R)
        points_G = cp.array(coordinate_G)
        points_B = cp.array(coordinate_B)

        # 为每个通道构建设计矩阵 A
        A_R = cp.c_[points_R[:, 0], points_R[:, 1], cp.ones(points_R.shape[0])]
        A_G = cp.c_[points_G[:, 0], points_G[:, 1], cp.ones(points_G.shape[0])]
        A_B = cp.c_[points_B[:, 0], points_B[:, 1], cp.ones(points_B.shape[0])]

        # 使用最小二乘法求解每个通道的平面拟合
        # lstsq 函数返回四个值：解、残差、秩和奇异值的数组
        C_R, _, _, _ = cp.linalg.lstsq(A_R, points_R[:, 2], rcond=None)
        C_G, _, _, _ = cp.linalg.lstsq(A_G, points_G[:, 2], rcond=None)
        C_B, _, _, _ = cp.linalg.lstsq(A_B, points_B[:, 2], rcond=None)

        # 解释结果
        print("R通道平面方程的系数为：A =", C_R[0], ", B =", C_R[1], ", C =", C_R[2])
        print("G通道平面方程的系数为：A =", C_G[0], ", B =", C_G[1], ", C =", C_G[2])
        print("B通道平面方程的系数为：A =", C_B[0], ", B =", C_B[1], ", C =", C_B[2])

        # 将通道转换为 NumPy 数组
        r_np = cp.array(r)
        g_np = cp.array(g)
        b_np = cp.array(b)

        # 创建一个新的图像数组来存储变换后的像素值
        transformed_r = cp.zeros_like(r_np)
        transformed_g = cp.zeros_like(g_np)
        transformed_b = cp.zeros_like(b_np)

        # 读取图像
        # image = cv2.imread('1719025648287.jpg')
        image = cv2.imread(image_path)
        image_gpu = cp.asarray(image)
        # 获取图像的高度和宽度
        height, width, _ = image_gpu.shape

        # 生成索引矩阵
        i_matrix, j_matrix = cp.meshgrid(cp.arange(height), cp.arange(width), indexing='ij')

        # 提取图像的各个通道
        B = image_gpu[:, :, 0]
        G = image_gpu[:, :, 1]
        R = image_gpu[:, :, 2]

        # 计算变换后的像素值
        transformed_r = cp.clip(C_R[0] * i_matrix + C_R[1] * j_matrix + R, 0, 255)
        transformed_g = cp.clip(C_G[0] * i_matrix + C_G[1] * j_matrix + G, 0, 255)
        transformed_b = cp.clip(C_B[0] * i_matrix + C_B[1] * j_matrix + B, 0, 255)

        # 将 GPU 数组转换为 CPU 数组以保存为图像
        transformed_r_cpu = cp.asnumpy(transformed_r)
        transformed_g_cpu = cp.asnumpy(transformed_g)
        transformed_b_cpu = cp.asnumpy(transformed_b)
        # 将 NumPy 数组转换回 PIL 图像
        transformed_r = Image.fromarray(transformed_r_cpu.astype('uint8'))
        transformed_g = Image.fromarray(transformed_g_cpu.astype('uint8'))
        transformed_b = Image.fromarray(transformed_b_cpu.astype('uint8'))

        # 分别保存三个通道的灰度图
        # 使用R通道创建灰度图
        gray_image_r = transformed_r
        gray_image_g = transformed_g
        gray_image_b = transformed_b
        # 保存灰度图

        # 合并通道
        transformed_image = Image.merge('RGB', (transformed_r, transformed_g, transformed_b))

        # 显示图像
        # transformed_image.show()
        transformed_image.save("transformed_image.jpg")

        # 创建一个新的图像数组来存储变换后平面拟合情况可视化像素值
        transformed_r_fitting = cp.zeros_like(r_np)
        transformed_g_fitting = cp.zeros_like(g_np)
        transformed_b_fitting = cp.zeros_like(b_np)

        # 计算变换后的像素值
        transformed_r_fitting = C_R[0] * i_matrix + C_R[1] * j_matrix + C_R[2]
        transformed_g_fitting = C_G[0] * i_matrix + C_G[1] * j_matrix + C_G[2]
        transformed_b_fitting = C_B[0] * i_matrix + C_B[1] * j_matrix + C_B[2]

        # 将 GPU 数组转换为 CPU 数组以保存为图像
        transformed_r_fitting_cpu = cp.asnumpy(transformed_r_fitting)
        transformed_g_fitting_cpu = cp.asnumpy(transformed_g_fitting)
        transformed_b_fitting_cpu = cp.asnumpy(transformed_b_fitting)

        # 将 NumPy 数组转换回 PIL 图像
        fitting_image_r_visible = Image.fromarray(transformed_r_fitting_cpu.astype('uint8'))
        fitting_image_g_visible = Image.fromarray(transformed_g_fitting_cpu.astype('uint8'))
        fitting_image_b_visible = Image.fromarray(transformed_b_fitting_cpu.astype('uint8'))

        # 合并通道
        fitting_image = Image.merge('RGB', (fitting_image_r_visible, fitting_image_g_visible, fitting_image_b_visible))
        # 显示图像
        # transformed_image.show()
        fitting_image.save("fitting_image.jpg")


        print('校准完成')


    def process_image(self):
        if not self.input_image_path or not self.output_directory:
            print("请先选择输入图像文件和输出文件夹。")
            return

        if self.input_image_path != self.inputpath_cache:
            real_path = self.cropper(self.input_image_path)
            self.inputpath_cache = self.input_image_path

        else:
            real_path = self.cropper(self.last_cropped_image_path)



            # 如果裁剪成功，更新上次裁剪的图像路径

        self.last_cropped_image_path = real_path

        # 获取用户输入的参数

        nChannel = int(self.nChannel_edit.text())
        nConv = int(self.nConv_edit.text())
        maxIter = int(self.maxIter_edit.text())
        minLabels = int(self.minLabels_edit.text())
        lr = float(self.lr_edit.text())
        output_r_path = os.path.join(self.output_directory, 'r_channel.jpeg')
        image = io.imread(real_path)
        img_r = image[:, :, 0].astype(np.uint8)



        img_denoised = cv2.fastNlMeansDenoising(img_r, h=3, templateWindowSize=5, searchWindowSize=7)
        # 计算原始图像中0和255占比
        low_percentile = 0.5  # 排除最低1%的像素
        high_percentile = 99  # 排除最高1%的像素
        '''
        # 不同场景下的参数组合
        # 案例1：轻微噪点
        low_percentile = 2
        high_percentile = 98

        # 案例2：严重椒盐噪声
        low_percentile = 5
        high_percentile = 95

        # 案例3：保留超暗/超亮细节
        low_percentile = 0.5
        high_percentile = 99.5

        '''
        min_val = np.percentile(img_denoised, low_percentile)
        max_val = np.percentile(img_denoised, high_percentile)
        print(max_val)
        print(min_val)
        # img_r = median(img_r, disk(3))
        img_clip = 255 * (img_denoised - min_val) / (max_val - min_val + 1e-8)  # 防止除以0
        img_clip = np.clip(img_clip, 0, 255).astype(np.uint8)


        if os.path.exists(output_r_path):
            counter = 1
            while os.path.exists(os.path.join(self.output_directory, f'r_channel_{counter}.jpeg')):
                counter += 1
            output_r_path = os.path.join(self.output_directory, f'r_channel_{counter}.jpeg')

        def smooth_extreme_regions(img, target_value=0, inpaint_radius=3):
            """
            将灰度为target_value的区域平滑为周围区域的均值
            :param img: 输入图像
            :param target_value: 需要平滑的灰度值（可以是0或255）
            :param inpaint_radius: 图像修复的半径（越大平滑效果越强）
            :return: 处理后的图像
            """
            # 创建掩模，标记灰度为target_value的像素
            mask = np.uint8(img == target_value) * 255

            # 使用inpaint进行图像修复
            img_inpainted = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)

            # cv2.imwrite(output_r_mask, mask )
            return img_inpainted

        img_smooth_0 = smooth_extreme_regions(img_clip, target_value=0, inpaint_radius=1)
        img_smooth_255 = smooth_extreme_regions(img_clip, target_value=255, inpaint_radius=3)
        img_smooth = np.maximum(img_smooth_0, img_smooth_255)


        cv2.imwrite(output_r_path, img_smooth)



        print(f"R通道图像已保存为 {output_r_path}")
        # 确保新文件不会被覆盖

        plt.subplot(1, 2, 1)
        io.imshow(image)
        plt.title('原始图片')

        plt.subplot(1, 2, 2)
        io.imshow(img_smooth)
        plt.title('R通道的图片')
        plt.show()
        #image_r为二维矩阵，单通道
        self.run_model(output_r_path, nChannel, nConv, maxIter, minLabels, lr)


    def run_model(self, output_r_path, nChannel, nConv, maxIter, minLabels, lr):
        losses = []  # 添加

        use_cuda = torch.cuda.is_available()
        #im为三通道的二维矩阵，每个通道都相等
        im = cv2.imread(output_r_path)

        # 将图像转换为PyTorch张量
        data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
        if use_cuda:
            data = data.cuda()
        data = Variable(data)

        #data.size(1)=3,（2）高度，
        visualize = 1
        # 初始化模型和优化器
        model = MyNet(data.size(1), nChannel, nConv).cuda()

        # 加载已存在的权重文件
        if self.weight_file_path and os.path.exists(self.weight_file_path):
            model.load_state_dict(torch.load(self.weight_file_path))
            print("加载选择的权重文件。")
            self.weight_file_path=''
        else:
            print("未选择权重文件，将使用默认权重。")

        weight_file = os.path.join(self.output_directory, 'model_weights.pth')
        if os.path.exists(weight_file):
            counter = 1
            while os.path.exists(os.path.join(self.output_directory, f'model_weights_{counter}.pth')):
                counter += 1
            weight_file = os.path.join(self.output_directory, f'model_weights_{counter}.pth')

        if use_cuda:
            model.cuda()
            print('cuda已使用')
        optimizer = optim.Adam(model.parameters(), lr=lr)
        def hue_to_rgb(hue):
            # HSL to RGB conversion logic
            if hue < 1 / 6:
                return np.array([1, hue * 6, 0])
            elif hue < 1 / 3:
                return np.array([(1 - (hue - 1 / 6) * 6), 1, 0])
            elif hue < 1 / 2:
                return np.array([0, 1, (hue - 1 / 3) * 6])
            elif hue < 2 / 3:
                return np.array([0, (1 - (hue - 1 / 2) * 6), 1])
            elif hue < 5 / 6:
                return np.array([(hue - 2 / 3) * 6, 0, 1])
            else:
                return np.array([1, 0, (1 - hue) * 6])


        hues = np.linspace(0, 1, nChannel, endpoint=False)  # 生成均匀分布的色相
        ori_colours = np.array([hue_to_rgb(hue) for hue in hues]) * 255
        label_colours = []

        # 将 ori_colours 切成三部分

        split_colours = np.array_split(ori_colours, 3)

        from itertools import zip_longest
        # 循环添加每个部分的元素
        for colors in zip_longest(*split_colours, fillvalue=None):
            for color in colors:
                if color is not None:  # 过滤掉填充的 None 值
                    label_colours.append(color)
        label_colours = np.array(label_colours)

        print(label_colours.astype(int))  # 打印固定的颜色

        # 定义损失函数
        loss_fn = torch.nn.CrossEntropyLoss().cuda()
        loss_hpy = torch.nn.L1Loss(reduction='mean').cuda()
        loss_hpz = torch.nn.L1Loss(reduction='mean').cuda()

        HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], nChannel)
        HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, nChannel)

        if use_cuda:
            HPy_target = HPy_target.cuda()
            HPz_target = HPz_target.cuda()

        # 训练模型
        for batch_idx in range(maxIter):
            optimizer.zero_grad()
            output = model(data)[0]

            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr}")


            output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
            outputHP = output.reshape((im.shape[0], im.shape[1], nChannel))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)
            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))


            # 统计标签频率
            unique, counts = np.unique(im_target, return_counts=True)
            label_counts = dict(zip(unique, counts))
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)


            #print(sorted_labels)
            # #[(1, 1621343), (29, 682256), (25, 165663), (3, 9740), (24, 237), (26, 79), (19, 30)]
            # 选择高频标签（例如，前8个）


            fixed_labels = [label for label, count in sorted_labels[:]]

            #print(fixed_labels)
            #[1, 29, 25, 3, 24, 26, 19]
            # 创建固定颜色映射
            fixed_colours = {fixed_labels[i]: label_colours[i % len(label_colours)] for i in range(len(fixed_labels))}

            print(fixed_colours)
            #{1: array([159,  19, 191]), 29: array([144, 131,  99]), 25: array([ 40, 246, 133]), 3: array([198,  79,  73]), 24: array([182, 244, 228]), 26: array([ 64,  58, 250]), 19: array([ 31, 197, 142])}



            if visualize:
                im_target_rgb = np.array([fixed_colours[c % nChannel] for c in im_target])
                im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

                desired_width = 1080  # 您希望的宽度
                desired_height = 720
                im_target_rgb_resized = cv2.resize(im_target_rgb, (desired_width, desired_height))
                cv2.imshow("output", im_target_rgb_resized)
                cv2.waitKey(10)

            # 损失计算
            loss = loss_fn(output, target) + (lhpy + lhpz)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(batch_idx, '/', maxIter, '|', 'label num:', nLabels, '| loss:', loss.item())
            if batch_idx % 100 == 0:
                torch.save(model.state_dict(), weight_file)
                print(f"权重已保存到 {weight_file}")

            if nLabels <= minLabels:
                print("nLabels", nLabels, "reached minLabels", minLabels, ".")
                break
        torch.save(model.state_dict(), weight_file)
        print(f"最终权重已保存到 {weight_file}")

        base_output_file = os.path.splitext(os.path.basename(self.input_image_path))[0]
        output_file = os.path.join(self.output_directory, base_output_file + '.png')

        # 检查文件是否存在，如果存在则添加编号
        if os.path.exists(output_file):
            counter = 1
            while os.path.exists(os.path.join(self.output_directory, f"{base_output_file}_{counter}.png")):
                counter += 1
            output_file = os.path.join(self.output_directory, f"{base_output_file}_{counter}.png")

        # 保存图像到用户选择的文件夹
        cv2.imwrite(output_file, im_target_rgb)  # Replace with processed image if applicable
        print(f"结果已保存到 {output_file}")



        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_directory, 'loss_curve.png'))
        plt.close()

        features = outputHP.reshape(-1, nChannel).cpu().detach().numpy()  # 特征矩阵 (H*W, nChannel)
        labels = im_target.flatten()  # 标签向量 (H*W,)

        # 随机采样（防止内存不足）
        sample_size = 100000  # 可根据显存调整
        if len(labels) > sample_size:
            indices = np.random.choice(len(labels), sample_size, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]
        else:
            features_sample = features
            labels_sample = labels

        # 计算评估指标
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        eval_results = {}
        try:
            if len(np.unique(labels_sample)) > 1:  # 至少需要2个类别
                eval_results['silhouette'] = silhouette_score(features_sample, labels_sample)
                eval_results['calinski_harabasz'] = calinski_harabasz_score(features_sample, labels_sample)
                eval_results['davies_bouldin'] = davies_bouldin_score(features_sample, labels_sample)
            else:
                print("警告：只有一个聚类，无法计算指标")
        except Exception as e:
            print(f"评估失败: {str(e)}")

        # 输出结果
        if eval_results:
            print("\n===== 聚类评估结果 =====")
            print(f"轮廓系数 (越高越好): {eval_results['silhouette']:.4f}")
            print(f"Calinski-Harabasz指数 (越高越好): {eval_results['calinski_harabasz']:.4f}")
            print(f"Davies-Bouldin指数 (越低越好): {eval_results['davies_bouldin']:.4f}")

            # 保存到文件
            eval_file = os.path.join(self.output_directory, 'cluster_metrics.txt')
            with open(eval_file, 'w') as f:
                f.write(f"Silhouette Score: {eval_results['silhouette']:.4f}\n")
                f.write(f"Calinski-Harabasz Index: {eval_results['calinski_harabasz']:.4f}\n")
                f.write(f"Davies-Bouldin Index: {eval_results['davies_bouldin']:.4f}\n")
        #
        # # 在保存结果后添加统计信息
        # unique_labels, counts = np.unique(im_target, return_counts=True)
        # print(f"分割区域数量: {len(unique_labels)}")
        # print("各区域像素占比:")
        # for label, count in zip(unique_labels, counts):
        #     print(f"标签 {label}: {count/im_target.size*100:.2f}%")
        # # 生成区域大小分布直方图
        # plt.figure(figsize=(10, 5))
        # plt.hist(counts, bins=20, log=True)
        # plt.title('Region Size Distribution')
        # plt.xlabel('Region Size (pixels)')
        # plt.ylabel('Frequency (log scale)')
        # plt.savefig(os.path.join(self.output_directory, 'region_distribution.png'))
        # plt.close()
class ResidualBlock(nn.Module):
    def __init__(self, nChannel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nChannel, nChannel, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(nChannel),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nChannel, nChannel, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(nChannel)
        )

    def forward(self, x):
        return F.leaky_relu(x + self.conv(x), 0.2)

class MyNet(nn.Module):
    def __init__(self, input_dim, nChannel, nConv):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_dim, nChannel, 3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(nChannel),
            nn.LeakyReLU(0.2)
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(nChannel) for _ in range(nConv - 1)]
        )

        self.final = nn.Sequential(
            nn.Conv2d(nChannel, nChannel, 1),
            nn.BatchNorm2d(nChannel),
            nn.LeakyReLU(0.2)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        return self.final(x)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



#高学习率可能更好，卷积核调1可能更好，SGD和ADM不确定