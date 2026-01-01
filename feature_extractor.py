import cv2
import numpy as np

class FeatureExtractor:

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError("图像读取失败，请检查路径是否正确")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    #1.颜色特征（颜色直方图）
    def color_histogram(self, color_space="RGB", bins=256):
        """
        计算颜色直方图（支持 RGB 三通道）
        :param color_space: 'RGB' or 'HSV'
        :param bins: 直方图维度
        :return:
            - 若 RGB: (hist_r, hist_g, hist_b) 每个形状为 (bins, 1)
            - 若 HSV: hist_h 形状为 (bins, 1)（当前仅返回 H 通道）
        """
        if color_space == "RGB":
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # 三通道分别统计
            hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
            hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])
            # 归一化（便于对比与显示）
            cv2.normalize(hist_r, hist_r)
            cv2.normalize(hist_g, hist_g)
            cv2.normalize(hist_b, hist_b)

            return hist_r, hist_g, hist_b

        elif color_space == "HSV":
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([img], [0], None, [bins], [0, 180])
            cv2.normalize(hist_h, hist_h)
            return hist_h

        else:
            raise ValueError("不支持的颜色空间")

    #2.频域特征
    def fourier_magnitude_spectrum(self, use_log=True):
        """
        计算傅里叶幅度谱（频域特征可视化）
        :param use_log: 是否使用对数增强显示
        :return: spectrum_img (uint8), low_high_energy_ratio (float)
        """
        img = self.gray.astype(np.float32) #FFT对浮点更稳定

        f = np.fft.fft2(img) #二维傅里叶变换
        fshift = np.fft.fftshift(f) #把低频成分移动到频谱中心
        magnitude = np.abs(fshift) #取复数幅值

        if use_log:
            magnitude = np.log1p(magnitude) #压缩动态范围，避免log(0)

        # 归一化到0~255便于显示
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        spectrum_img = mag_norm.astype(np.uint8) #转成uint8

        # 一个简单的“频域描述”：低频能量/高频能量比值
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2 #频谱中心点
        r = int(min(h, w) * 0.10) #低频半径，r越大，低频区域越大
        yy, xx = np.ogrid[:h, :w] #生成行列索引网格
        mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

        low_energy = float(magnitude[mask_low].sum()) #低频能量=中心圆区域能量总和
        high_energy = float(magnitude[~mask_low].sum()) + 1e-6 #高频能量 = 除中心外的能量总和
        ratio = low_energy / high_energy

        return spectrum_img, ratio

    #3.边缘特征（Canny）
    def canny_edges(self, low_thresh=50, high_thresh=150, blur_ksize=5):
        """
        Canny 边缘检测 + 边缘特征描述
        :param low_thresh: Canny 低阈值
        :param high_thresh: Canny 高阈值
        :param blur_ksize: 高斯滤波核大小（奇数）
        :return: edges(uint8), edge_density(float), edge_pixels(int)
        """
        if blur_ksize and blur_ksize >= 3: #blur≥3才做高斯滤波
            blurred = cv2.GaussianBlur(self.gray, (blur_ksize, blur_ksize), 0)
        else:
            blurred = self.gray

        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        edge_pixels = int(np.count_nonzero(edges)) #统计边缘像素数
        total_pixels = int(edges.size)
        edge_density = edge_pixels / (total_pixels + 1e-6)  # 边缘点占全图像素比例

        return edges, edge_density, edge_pixels

    #4.纹理特征（LBP 简化版）
    def lbp_texture(self):
        """
        计算 LBP 纹理特征（简化版）
        :return: lbp_image
        """
        lbp = np.zeros_like(self.gray) #建一个和灰度图同尺寸的lbp输出图
        rows, cols = self.gray.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1): #避开边界（因为LBP要访问8邻域，边界像素没有完整邻域）
                center = self.gray[i, j] #当前像素为中心点
                binary_string = '' #用来拼8位二进制编码
                #按固定顺序比较 8 邻域像素与中心像素大小关系，大于中心为1，反之为0，得到8位二进制编码
                binary_string += '1' if self.gray[i - 1, j - 1] > center else '0'
                binary_string += '1' if self.gray[i - 1, j] > center else '0'
                binary_string += '1' if self.gray[i - 1, j + 1] > center else '0'
                binary_string += '1' if self.gray[i, j + 1] > center else '0'
                binary_string += '1' if self.gray[i + 1, j + 1] > center else '0'
                binary_string += '1' if self.gray[i + 1, j] > center else '0'
                binary_string += '1' if self.gray[i + 1, j - 1] > center else '0'
                binary_string += '1' if self.gray[i, j - 1] > center else '0'
                #把二进制字符串转成0~255的整数，作为该像素的LBP值
                lbp[i, j] = int(binary_string, 2)

        return lbp

    def lbp_histogram(self):
        """
        计算 LBP 纹理直方图（用于特征描述）
        """
        lbp = self.lbp_texture() #先计算LBP图
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256)) #统计0~255各LBP值的出现频率
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6) #归一化成概率分布
        return hist