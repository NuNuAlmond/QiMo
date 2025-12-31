import sys
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog,
    QComboBox, QTextEdit, QSpinBox, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from feature_extractor import FeatureExtractor


class FeatureGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像特征提取系统（Python + OpenCV）")
        self.resize(900, 600)

        self.image_path = None
        self.extractor = None

        self.btn_load = None
        self.btn_extract = None
        self.feature_box = None
        self.image_label = None
        self.info_box = None

        self.init_ui()

    def init_ui(self):
        # ===== 左侧：按钮和选项 =====
        self.btn_load = QPushButton("加载图像")
        self.btn_extract = QPushButton("提取特征")

        self.feature_box = QComboBox()
        self.feature_box.addItems([
            "颜色特征（HSV直方图）",
            "边缘特征（Canny）",
            "频域特征（傅里叶幅度谱）",
            "纹理特征（LBP）",
            "几何特征（ORB关键点）",
            "角点检测（Harris）",
        ])

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.feature_box)

        # ===== Canny 参数组（默认先隐藏）=====
        self.canny_group = QGroupBox("Canny 参数")
        canny_layout = QVBoxLayout()

        # low threshold
        self.spin_canny_low = QSpinBox()
        self.spin_canny_low.setRange(0, 500)
        self.spin_canny_low.setValue(50)
        self.spin_canny_low.setPrefix("low = ")

        # high threshold
        self.spin_canny_high = QSpinBox()
        self.spin_canny_high.setRange(0, 500)
        self.spin_canny_high.setValue(150)
        self.spin_canny_high.setPrefix("high = ")

        # blur kernel size (odd)
        self.spin_canny_blur = QSpinBox()
        self.spin_canny_blur.setRange(1, 31)
        self.spin_canny_blur.setSingleStep(2)  # 步长 2，天然保持奇数
        self.spin_canny_blur.setValue(5)
        self.spin_canny_blur.setPrefix("blur = ")

        canny_layout.addWidget(self.spin_canny_low)
        canny_layout.addWidget(self.spin_canny_high)
        canny_layout.addWidget(self.spin_canny_blur)

        self.canny_group.setLayout(canny_layout)
        self.canny_group.setVisible(False)  # 默认隐藏

        left_layout.addWidget(self.canny_group)

        left_layout.addWidget(self.btn_extract)
        left_layout.addStretch()

        # ===== 右侧：图像显示 =====
        self.image_label = QLabel("请加载一张图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)

        # 特征描述文本框
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setFixedHeight(150)
        self.info_box.setText("特征描述信息将在此显示")

        right_layout.addWidget(self.info_box)

        # ===== 主布局 =====
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 8)

        self.setLayout(main_layout)

        # ===== 信号绑定 =====
        self.btn_load.clicked.connect(self.load_image)
        self.btn_extract.clicked.connect(self.extract_feature)
        self.feature_box.currentTextChanged.connect(self.on_feature_changed)

    # 只有选择 Canny 时显示参数组
    def on_feature_changed(self, text: str):
        if "Canny" in text or "边缘" in text:
            self.canny_group.setVisible(True)
        else:
            self.canny_group.setVisible(False)

    # =============================
    # 功能函数
    # =============================
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            "",
            "Image Files (*.jpg *.png *.bmp)"
        )

        if file_path:
            self.image_path = file_path
            self.extractor = FeatureExtractor(file_path)
            self.show_image(self.extractor.image)

    def extract_feature(self):
        if self.extractor is None:
            return

        feature_type = self.feature_box.currentText()

        if "颜色" in feature_type:
            hist = self.extractor.color_histogram()
            hist_img = self.hist_to_image(hist)
            self.show_image(hist_img)

            mean = np.mean(hist)
            std = np.std(hist)

            self.info_box.setText(
                "【颜色特征描述】\n"
                "颜色空间：HSV\n"
                f"直方图均值：{mean:.4f}\n"
                f"直方图标准差：{std:.4f}\n"
                "说明：用于描述图像整体颜色分布情况。"
            )

        elif "傅里叶" in feature_type:
            spectrum_img, ratio = self.extractor.fourier_magnitude_spectrum(use_log=True)
            self.show_image(spectrum_img, gray=True)

            self.info_box.setText(
                "【频域特征描述】\n"
                "方法：2D FFT（傅里叶变换）\n"
                "显示：幅度谱（对数增强）\n"
                f"低频/高频能量比：{ratio:.4f}\n"
                "说明：低频反映整体结构，高频反映边缘与细节纹理。"
            )

        elif "Canny" in feature_type or "边缘" in feature_type:
            low = int(self.spin_canny_low.value())
            high = int(self.spin_canny_high.value())
            blur = int(self.spin_canny_blur.value())

            # 保证 high > low（防止用户调反）
            if high <= low:
                high = low + 1
                self.spin_canny_high.setValue(high)

            # blur 保持奇数（spin 已经步长 2，但保险一下）
            if blur % 2 == 0:
                blur += 1
                self.spin_canny_blur.setValue(blur)

            edges, density, edge_pixels = self.extractor.canny_edges(
                low_thresh=low, high_thresh=high, blur_ksize=blur
            )
            self.show_image(edges, gray=True)

            self.info_box.setText(
                "【边缘特征描述】\n"
                "方法：Canny 边缘检测\n"
                f"参数：low={low}, high={high}, blur={blur}\n"
                f"边缘像素数：{edge_pixels}\n"
                f"边缘密度（占比）：{density:.4f}\n"
                "说明：阈值越高，边缘更少更干净；阈值越低，边缘更丰富但可能噪声更多。"

            )

        elif "纹理" in feature_type:
            lbp = self.extractor.lbp_texture()
            self.show_image(lbp, gray=True)

            hist = self.extractor.lbp_histogram()

            self.info_box.setText(
                "【纹理特征描述】\n"
                "方法：LBP（局部二值模式）\n"
                f"直方图维度：{len(hist)}\n"
                f"最大纹理响应值：{np.max(hist):.4f}\n"
                "说明：LBP 用于描述图像局部纹理结构特征。"
            )

        elif "ORB" in feature_type:
            keypoints, descriptors = self.extractor.orb_features()
            kp_img = self.extractor.draw_keypoints(keypoints)
            self.show_image(kp_img)

            desc_dim = descriptors.shape[1] if descriptors is not None else 0

            self.info_box.setText(
                "【几何特征描述】\n"
                "特征类型：ORB\n"
                f"关键点数量：{len(keypoints)}\n"
                f"描述子维度：{desc_dim}\n"
                "说明：关键点数量反映图像结构复杂程度。"
            )

        elif "Harris" in feature_type:
            corners_img, corner_count, max_resp = self.extractor.harris_corners(
                block_size=2, ksize=3, k=0.04, thresh=0.01
            )
            self.show_image(corners_img)

            self.info_box.setText(
                "【角点特征描述】\n"
                "方法：Harris Corner Detector\n"
                "参数：block_size=2, ksize=3, k=0.04, thresh=0.01\n"
                f"角点数量：{corner_count}\n"
                f"最大响应值：{max_resp:.4f}\n"
                "说明：角点通常位于结构变化明显的位置，可用于匹配与识别。"
            )

    # =============================
    # 工具函数
    # =============================
    def show_image(self, img, gray=False):
        if gray:
            qimg = QImage(
                img.data, img.shape[1], img.shape[0],
                img.strides[0], QImage.Format_Grayscale8
            )
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],
                img_rgb.strides[0], QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    @staticmethod
    def hist_to_image(hist):
        hist = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)
        hist_img = np.zeros((300, 256, 3), dtype=np.uint8)

        for x in range(256):
            cv2.line(
                hist_img,
                (x, 300),
                (x, 300 - int(hist[x])),
                (255, 255, 255),
                1
            )
        return hist_img

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FeatureGUI()
    win.show()
    sys.exit(app.exec_())