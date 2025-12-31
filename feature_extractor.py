import cv2
import numpy as np

class FeatureExtractor:
    """
    图像特征提取类
    支持：颜色特征、频域特征、边缘特征、纹理特征、几何（关键点）特征、角点特征
    """

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError("图像读取失败，请检查路径是否正确")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # =========================
    # 1. 颜色特征（颜色直方图）
    # =========================
    def color_histogram(self, color_space="HSV", bins=256):
        """
        计算颜色直方图
        :param color_space: 'RGB' or 'HSV'
        :param bins: 直方图维度
        :return: histogram
        """
        if color_space == "RGB":
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        elif color_space == "HSV":
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError("不支持的颜色空间")

        hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
        cv2.normalize(hist, hist)

        return hist

    # =========================
    # 2. 频域特征
    # =========================
    def fourier_magnitude_spectrum(self, use_log=True):
        """
        计算傅里叶幅度谱（频域特征可视化）
        :param use_log: 是否使用对数增强显示
        :return: spectrum_img (uint8), low_high_energy_ratio (float)
        """
        img = self.gray.astype(np.float32)

        # 2D FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        if use_log:
            magnitude = np.log1p(magnitude)

        # 归一化到 0-255 便于显示
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        spectrum_img = mag_norm.astype(np.uint8)

        # 一个简单的“频域描述”：低频能量 / 高频能量比值
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        r = int(min(h, w) * 0.10)  # 低频半径（可调 0.05~0.2）
        yy, xx = np.ogrid[:h, :w]
        mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

        low_energy = float(magnitude[mask_low].sum())
        high_energy = float(magnitude[~mask_low].sum()) + 1e-6
        ratio = low_energy / high_energy

        return spectrum_img, ratio

    # =========================
    # 3. 边缘特征（Canny）
    # =========================
    def canny_edges(self, low_thresh=50, high_thresh=150, blur_ksize=5):
        """
        Canny 边缘检测 + 边缘特征描述
        :param low_thresh: Canny 低阈值
        :param high_thresh: Canny 高阈值
        :param blur_ksize: 高斯滤波核大小（奇数）
        :return: edges(uint8), edge_density(float), edge_pixels(int)
        """
        if blur_ksize and blur_ksize >= 3:
            blurred = cv2.GaussianBlur(self.gray, (blur_ksize, blur_ksize), 0)
        else:
            blurred = self.gray

        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        edge_pixels = int(np.count_nonzero(edges))
        total_pixels = int(edges.size)
        edge_density = edge_pixels / (total_pixels + 1e-6)  # 边缘像素占比（0~1）

        return edges, edge_density, edge_pixels

    # =========================
    # 4. 纹理特征（LBP 简化版）
    # =========================
    def lbp_texture(self):
        """
        计算 LBP 纹理特征（简化版）
        :return: lbp_image
        """
        lbp = np.zeros_like(self.gray)
        rows, cols = self.gray.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = self.gray[i, j]
                binary_string = ''

                binary_string += '1' if self.gray[i - 1, j - 1] > center else '0'
                binary_string += '1' if self.gray[i - 1, j] > center else '0'
                binary_string += '1' if self.gray[i - 1, j + 1] > center else '0'
                binary_string += '1' if self.gray[i, j + 1] > center else '0'
                binary_string += '1' if self.gray[i + 1, j + 1] > center else '0'
                binary_string += '1' if self.gray[i + 1, j] > center else '0'
                binary_string += '1' if self.gray[i + 1, j - 1] > center else '0'
                binary_string += '1' if self.gray[i, j - 1] > center else '0'

                lbp[i, j] = int(binary_string, 2)

        return lbp

    def lbp_histogram(self):
        """
        计算 LBP 纹理直方图（用于特征描述）
        """
        lbp = self.lbp_texture()
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    # =========================
    # 5. 几何特征（ORB / SIFT）
    # =========================
    def orb_features(self, max_features=500):
        """
        ORB 特征提取
        :return: keypoints, descriptors
        """
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints, descriptors = orb.detectAndCompute(self.gray, None)
        return keypoints, descriptors

    def draw_keypoints(self, keypoints):
        """
        可视化关键点
        """
        return cv2.drawKeypoints(
            self.image,
            keypoints,
            None,
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )

    # =========================
    # 6. 角点特征（Harris）
    # =========================
    def harris_corners(self, block_size=2, ksize=3, k=0.04, thresh=0.01):
        """
        Harris 角点检测
        :param block_size: 邻域大小
        :param ksize: Sobel 算子孔径
        :param k: Harris 参数，一般 0.04~0.06
        :param thresh: 阈值比例（相对最大响应）
        :return: corners_img(BGR), corner_count(int), max_response(float)
        """
        gray_f = np.float32(self.gray)
        dst = cv2.cornerHarris(gray_f, block_size, ksize, k)

        # 膨胀增强角点区域
        dst_dilated = cv2.dilate(dst, None)

        # 阈值判定
        threshold_value = thresh * dst_dilated.max()
        corners_mask = dst_dilated > threshold_value
        corner_count = int(np.count_nonzero(corners_mask))
        max_response = float(dst_dilated.max())

        # 在原图上标记角点
        corners_img = self.image.copy()
        corners_img[corners_mask] = [0, 0, 255]  # 标红角点（BGR）

        return corners_img, corner_count, max_response


