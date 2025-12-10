import os
import cv2
# Use for Resizing images to 224x224
# Path to your dataset
dataset_path = r"C:\Users\USER\Desktop\CNNmodel\rawdata\dataset_add\valid"
target_size = (224, 224)  # width, height

# Loop through each class folder
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")

    # Loop through each image in the class folder
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        # Resize to 224x224
        resized_img = cv2.resize(img, target_size)

        # Save back (overwrite)
        cv2.imwrite(img_path, resized_img)

print("All images resized to 224x224!")
