import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def bai_1():
    img = cv2.imread('canho.jpg')
    if img is None: return print("Khong tim thay canho.jpg")
    img = cv2.resize(img, (224, 224))
    flipped = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0
    plt.imshow(norm, cmap='gray')
    plt.title("Bai 1: Can ho")
    plt.show()

def bai_2():
    img = cv2.imread('oto.jpg')
    if img is None: return print("Khong tim thay oto.jpg")
    img = cv2.resize(img, (224, 224))
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    display_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB) / 255.0
    plt.imshow(display_img)
    plt.title("Bai 2: Oto")
    plt.show()

def bai_3():
    img = cv2.imread('traicay.jpg')
    if img is None: return print("Khong tim thay traicay.jpg")
    img_rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    for i in range(9):
        angle = random.randint(-20, 20)
        M = cv2.getRotationMatrix2D((112, 112), angle, 1.1)
        aug = cv2.warpAffine(img_rgb, M, (224, 224))
        plt.subplot(3, 3, i+1)
        plt.imshow(aug / 255.0)
        plt.axis('off')
    plt.suptitle("Bai 3: Grid 3x3 Trai cay")
    plt.show()


def bai_4():
    img = cv2.imread('noithat.jpg')
    if img is None: return print("Khong tim thay noithat.jpg")
    img_res = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    aug1 = cv2.flip(img_rgb, 1)
    aug2 = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(10, 4))
    imgs = [img_rgb, aug1, aug2]
    titles = ["Goc", "Flip", "Gray"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(imgs[i], cmap='gray' if i==2 else None)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    print("--- CHUONG TRINH DANG BAT DAU ---")
    try:
        print("Dang chay Bai 1...")
        bai_1()
        print("Dang chay Bai 2...")
        bai_2()
        print("Dang chay Bai 3...")
        bai_3()
        print("Dang chay Bai 4...")
        bai_4()
    except Exception as e:
        print(f"Co loi xay ra: {e}")
    print("--- KET THUC ---")