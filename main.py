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

#定义一个窗口类，继承自QWidget，用来作为程序的主窗口和图形界面容器
class FeatureGUI(QWidget):
    #FeatureGUI窗口类的构造函数
    def __init__(self):
        super().__init__() #调用父类QWidget的构造函数
        self.setWindowTitle("图像特征提取系统")
        self.resize(900, 600)

        self.image_path = None
        self.extractor = None

        self.btn_load = None
        self.btn_extract = None
        self.feature_box = None
        self.image_label = None
        self.info_box = None

        self.init_ui() #把界面搭建过程放进独立函数init_ui()

    def init_ui(self):
        #左侧：按钮和选项
        self.btn_load = QPushButton("加载图像")
        self.btn_extract = QPushButton("提取特征")

        self.feature_box = QComboBox()
        self.feature_box.addItems([
            "颜色特征（RGB直方图）",
            "边缘特征（Canny）",
            "频域特征（傅里叶变换）",
            "纹理特征（LBP）"
        ])

        left_layout = QVBoxLayout() #左侧布局采用竖直排列
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.feature_box)

        #Canny参数组（默认先隐藏）
        self.canny_group = QGroupBox("Canny参数")
        canny_layout = QVBoxLayout() #容器内部竖直布局
        #低阈值
        self.spin_canny_low = QSpinBox() #只能输入整数，QSpinBox整数输入控件
        self.spin_canny_low.setRange(0, 500)
        self.spin_canny_low.setValue(50)
        self.spin_canny_low.setPrefix("low = ") #setPrefix显示前缀
        #高阈值
        self.spin_canny_high = QSpinBox()
        self.spin_canny_high.setRange(0, 500)
        self.spin_canny_high.setValue(150)
        self.spin_canny_high.setPrefix("high = ")
        #高斯滤波核大小
        self.spin_canny_blur = QSpinBox()
        self.spin_canny_blur.setRange(1, 31)
        self.spin_canny_blur.setSingleStep(2) #步长2，保持奇数
        self.spin_canny_blur.setValue(5)
        self.spin_canny_blur.setPrefix("blur = ")
        #把3个输入框加入Canny内部布局
        canny_layout.addWidget(self.spin_canny_low)
        canny_layout.addWidget(self.spin_canny_high)
        canny_layout.addWidget(self.spin_canny_blur)

        self.canny_group.setLayout(canny_layout)#把布局装进groupbox
        self.canny_group.setVisible(False) #默认隐藏

        left_layout.addWidget(self.canny_group)
        left_layout.addWidget(self.btn_extract)
        left_layout.addStretch() #把上面的控件顶到上方，下面留空

        #右侧：图像显示和文本框
        self.image_label = QLabel("请加载一张图像")
        self.image_label.setAlignment(Qt.AlignCenter) #居中
        self.image_label.setStyleSheet("border: 1px solid gray") #灰色边框

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)
        #特征描述文本框
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True) #只显示输出，用户不能编辑
        self.info_box.setFixedHeight(150)
        self.info_box.setText("特征描述信息将在此显示")

        right_layout.addWidget(self.info_box)

        #主布局：左+右
        main_layout = QHBoxLayout() #整体横向布局
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 8)
        self.setLayout(main_layout) #把这个主布局设置为窗口的布局

        #信号绑定
        self.btn_load.clicked.connect(self.load_image)
        self.btn_extract.clicked.connect(self.extract_feature)
        self.feature_box.currentTextChanged.connect(self.on_feature_changed)

    #点击“加载图像”按钮时，Qt会自动调用load_image()
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName( #弹出文件选择对话框
            self,
            "选择图像",
            "D:\PyCharmProject\QiMo\images_data",
            "Image Files (*.jpg *.png *.bmp)"
        )
        if file_path: #如果用户点了取消，file_path是空字符串""
            self.image_path = file_path
            self.extractor = FeatureExtractor(file_path)
            self.show_image(self.extractor.image)

    #点击“提取特征”按钮是，调用extract_feature()
    def extract_feature(self):
        if self.extractor is None: #判断是否已加载图像
            return

        feature_type = self.feature_box.currentText() #获取当前下拉框的字符串

        if "颜色" in feature_type:
            hist_r, hist_g, hist_b = self.extractor.color_histogram(color_space="RGB", bins=256)
            hist_img = self.rgb_hist_to_image(hist_r, hist_g, hist_b) #把三条直方图曲线画到一张numpy图像上
            self.show_image(hist_img)
            #做一个简单的定量描述，三通道mean均值与std标准差
            r_mean, r_std = float(np.mean(hist_r)), float(np.std(hist_r))
            g_mean, g_std = float(np.mean(hist_g)), float(np.std(hist_g))
            b_mean, b_std = float(np.mean(hist_b)), float(np.std(hist_b))
            self.info_box.setText(
                "【颜色特征描述】\n"
                "颜色空间：RGB\n"
                f"R直方图：均值={r_mean:.4f}, 标准差={r_std:.4f}\n"
                f"G直方图：均值={g_mean:.4f}, 标准差={g_std:.4f}\n"
                f"B直方图：均值={b_mean:.4f}, 标准差={b_std:.4f}\n"
                "说明：RGB三通道直方图用于描述图像颜色分布与各通道强度差异。标准差越大说明直方图更“尖锐/起伏大”，颜色分布更集中或差异更明显；标准差越小说明分布更“平缓”，颜色更均匀或更分散。"
            )

        elif "傅里叶" in feature_type:
            spectrum_img, ratio = self.extractor.fourier_magnitude_spectrum(use_log=True)
            self.show_image(spectrum_img, gray=True) #gray=True按灰度格式显示
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
            #保证high>low
            if high <= low:
                high = low + 1
                self.spin_canny_high.setValue(high)
            #blur保持奇数
            if blur % 2 == 0:
                blur += 1
                self.spin_canny_blur.setValue(blur)
            edges, density, edge_pixels = self.extractor.canny_edges(low_thresh=low, high_thresh=high, blur_ksize=blur)
            self.show_image(edges, gray=True)
            self.info_box.setText(
                "【边缘特征描述】\n"
                "方法：Canny边缘检测\n"
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
                "说明：LBP用于描述图像局部纹理结构特征。"
            )

    # 只有选择Canny时显示参数组
    def on_feature_changed(self, text: str):
        if "Canny" in text or "边缘" in text:
            self.canny_group.setVisible(True)
        else:
            self.canny_group.setVisible(False)

    #把OpenCV/Numpy图像显示到Qt窗口（QLabel）上
    def show_image(self, img, gray=False):
        if gray: #如果是灰度图，直接构造QImage
            qimg = QImage(img.data, img.shape[1], img.shape[0],img.strides[0], QImage.Format_Grayscale8)
            #img.data：numpy底层内存指针；img.shape[1]：宽度；img.shape[0]：高度；img.strides[0]：每一行占用的字节数；QImage.Format_Grayscale8：8位灰度图格式
        else: #如果是彩色图，先BGR转RGB，再构造QImage
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],img_rgb.strides[0], QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg) #QImage转QPixmap，QLabel更适合显示pixmap
        pixmap = pixmap.scaled(self.image_label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        #self.image_label.size()：目标显示区域的大小；Qt.KeepAspectRatio：保持宽高比，不拉伸变形；Qt.SmoothTransformation：高质量缩放
        self.image_label.setPixmap(pixmap) #把缩放后的pixmap设置到QLabel

    @staticmethod
    def rgb_hist_to_image(hist_r, hist_g, hist_b, height=300, width=256): #将RGB三通道直方图绘制成可显示的图像（BGR格式）
        # 统一归一化到0~255，并拉平成一维
        r = cv2.normalize(hist_r, None, 0, 255, cv2.NORM_MINMAX).ravel()
        g = cv2.normalize(hist_g, None, 0, 255, cv2.NORM_MINMAX).ravel()
        b = cv2.normalize(hist_b, None, 0, 255, cv2.NORM_MINMAX).ravel()
        hist_img = np.zeros((height, width, 3), dtype=np.uint8) #创建一张全黑的画布
        #遍历每一个bin，并画三条竖线（R/G/B各一条）
        for x in range(width):
            #OpenCV画线用BGR：红线=(0,0,255), 绿线=(0,255,0), 蓝线=(255,0,0)
            cv2.line(hist_img, (x, height), (x, height - int(r[x])), (0, 0, 255), 1)
            cv2.line(hist_img, (x, height), (x, height - int(g[x])), (0, 255, 0), 1)
            cv2.line(hist_img, (x, height), (x, height - int(b[x])), (255, 0, 0), 1)
        return hist_img

#整个PyQt5程序入口
if __name__ == "__main__": #防止该文件被导入时也自动弹出窗口、自动运行程序
    app = QApplication(sys.argv) #创建Qt应用对象
    win = FeatureGUI() #创建主窗口对象
    win.show() #显示窗口
    sys.exit(app.exec_()) #启动事件循环，并确保正常退出
