import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import cupy as cp
import os
import matplotlib
matplotlib.use('Agg')

# 函数将平面拟合为圆内的像素
def fit_plane(points):
    A = cp.c_[points[:, 0], points[:, 1], cp.ones(points.shape[0])]
    C, _, _, _ = cp.linalg.lstsq(A, points[:, 2], rcond=None)
    return C


# 用于计算拟合的均方误差的函数
def mse_plane(points, C):
    A = cp.c_[points[:, 0], points[:, 1], cp.ones(points.shape[0])]
    predicted = A @ C
    mse = cp.mean((predicted - points[:, 2]) ** 2)
    return mse


# 获取圆内点的函数
def get_circle_points(center, radius, image):
    x_center, y_center = center
    indices = cp.indices((image.shape[1], image.shape[0])).transpose(1, 2, 0).reshape(-1, 2)
    distances = cp.sqrt((indices[:, 0] - x_center) ** 2 + (indices[:, 1] - y_center) ** 2)
    mask = distances <= radius
    points = indices[mask]
    z_values = image[points[:, 1], points[:, 0]]
    return cp.c_[points, z_values]


# 获取单像素厚度的圆环上的点
def get_annulus_points(center, inner_radius, outer_radius, image):
    x_center, y_center = center
    indices = cp.indices((image.shape[1], image.shape[0])).transpose(1, 2, 0).reshape(-1, 2)
    distances = cp.sqrt((indices[:, 0] - x_center) ** 2 + (indices[:, 1] - y_center) ** 2)
    mask = (distances > inner_radius) & (distances <= outer_radius)
    points = indices[mask]
    z_values = image[points[:, 1], points[:, 0]]
    return cp.c_[points, z_values]


# 用于判断外围像素RGB值与圆内像素RGB值变化的函数
def is_significant_change(inner_points, outer_points, threshold):
    inner_mean = cp.mean(inner_points, axis=0)
    outer_mean = cp.mean(outer_points, axis=0)
    change = cp.linalg.norm(inner_mean - outer_mean)
    return change > threshold


# 判断新圆是否与已有圆重叠
def is_overlapping(new_center, new_radius, base_regions):
    x_new, y_new = new_center
    for x_center, y_center, radius in base_regions:
        distance = cp.sqrt((x_new - x_center) ** 2 + (y_new - y_center) ** 2)
        if distance < new_radius + radius:
            return True
    return False


# 处理图像的主函数
def process_image(image, max_radius=100, min_radius=40, rgb_threshold=1, tolerance=1e-3, min_baseline_points=4):
    height, width, _ = image.shape
    avg_rgb = cp.mean(image.reshape(-1, 3), axis=0)  # 防止圆圈选在边缘黑色区域
    base_regions = []
    while len(base_regions) < min_baseline_points:
        # 随机选择一个点
        x_center = random.randint(0, width - 1)
        y_center = random.randint(0, height - 1)
        ## 防止圆圈选在边缘黑色区域
        center_rgb = image[y_center, x_center]
        # 计算中心点与全图平均RGB值的差距
        rgb_diff = cp.linalg.norm(center_rgb - avg_rgb, ord=1)
        # 如果差距过大则放弃该点
        if rgb_diff > 50:
            continue
        radius = 5  # 从小半径开始
        prev_C = None
        while radius <= max_radius:
            points = get_circle_points((x_center, y_center), radius, image)
            C = fit_plane(points)
            if prev_C is not None:
                mse_change = mse_plane(points, C) - mse_plane(points, prev_C)
                if mse_change > tolerance:
                    break
            if radius > 5:
                inner_points = get_circle_points((x_center, y_center), radius - 5, image)
                outer_points = get_annulus_points((x_center, y_center), radius - 5, radius, image)
                if is_significant_change(inner_points[:, 2:], outer_points[:, 2:], rgb_threshold) and radius <= min_radius:
                    print("give up this point, now get " + str(len(base_regions)) + " point")
                    break
                elif is_significant_change(inner_points[:, 2:], outer_points[:, 2:], rgb_threshold) and radius > min_radius:
                    if not is_overlapping((x_center, y_center), radius - 5, base_regions):
                        base_regions.append((x_center, y_center, radius - 5))
                        print("Currently available: " + str(len(base_regions)) + " point")
                    else:
                        break
            prev_C = C
            radius += 5  # 增加半径
    return base_regions


# 筛选出可能随机选在错误像素的圆圈
def robust_mean(arrays, threshold=2.5):
    data = cp.array(arrays)
    medians = cp.median(data, axis=0)
    deviations = cp.abs(data - medians)
    mad = cp.median(deviations, axis=0)
    mad_std = mad * 1.4826
    is_outlier = deviations > (threshold * mad_std)
    data[is_outlier] = cp.nan
    means = cp.nanmean(data, axis=0)
    return means


# 处理图片路径和背景色的RGB值
def unify_background(image_path, background_rgb):
    image = Image.open(image_path)
    image_rgba = image.convert('RGBA')
    width, height = image.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = image_rgba.getpixel((x, y))
            if abs(r - background_rgb[0]) < 5 and abs(g - background_rgb[1]) < 15 and abs(b - background_rgb[2]) < 15:
                image_rgba.putpixel((x, y), (int(background_rgb[0]), int(background_rgb[1]), int(background_rgb[2]), a))
    image_rgb = image_rgba.convert('RGB')
    image_rgb.save('processed_image.jpg')
    # image_rgb.show()


