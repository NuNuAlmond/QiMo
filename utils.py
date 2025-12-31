import cv2
import matplotlib.pyplot as plt


def show_image(title, image, is_gray=False):
    plt.figure(figsize=(6, 6))
    if is_gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_histogram(hist, title="Histogram"):
    plt.figure()
    plt.plot(hist)
    plt.title(title)
    plt.show()
