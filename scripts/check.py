import cv2
import matplotlib.pyplot as plt
import os

gesture_folder = r"C:\Users\USER\Desktop\CNNmodel\rawdata\dataset\train\fist"
sample_images = os.listdir(gesture_folder)[:5]

for img_name in sample_images:
    img = cv2.imread(os.path.join(gesture_folder, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
