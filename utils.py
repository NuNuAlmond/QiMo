import cv2
import matplotlib.pyplot as plt #导入matplotlib的绘图接口，用于弹出窗口显示图片/曲线

def show_image(title, image, is_gray=False): #在一个窗口中显示图像，是否按灰度图显示
    plt.figure(figsize=(6, 6)) #新建一个matplotlib绘图窗口figure
    if is_gray: #如果is_gray=True，按灰度图显示
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off') #关闭x/y坐标轴刻度和边框
    plt.show() #弹出窗口并显示

def show_histogram(hist, title="Histogram"): #显示直方图/曲线
    plt.figure()
    plt.plot(hist)
    plt.title(title)
    plt.show()
