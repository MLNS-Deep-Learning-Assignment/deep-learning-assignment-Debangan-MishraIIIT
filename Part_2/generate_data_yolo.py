import random
import pickle
from torchvision import datasets
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

mnist_data = datasets.MNIST(root="./data", train=True, download=True)
num_samples = 30000
image_width = 168
image_height = 40
horizontal_spacing = 1
overlap_prob = 0.65  # Probability of overlap between digits

output_dir = "synthetic_data2"
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

def normalize(value, max_value):
    return value / max_value

for idx in tqdm(range(num_samples), desc="Generating Images"):

    composite_image = Image.new("L", (image_width, image_height), color=0)
    annotation_lines = []

    num_digits = random.choices([3, 4, 5], weights=[0.05, 0.9, 0.05])[0]
    digits = random.choices(range(len(mnist_data)), k=num_digits)
    current_x = 0

    for digit_idx in digits:
        digit_image, label = mnist_data[digit_idx]

        scale = random.uniform(0.8, 1.2)
        new_width = int(digit_image.width * scale)
        new_height = int(digit_image.height * scale)
        digit_image = digit_image.resize((new_width, new_height))
        max_y = image_height - new_height
        y_offset = random.randint(0, max_y)

        # Decide whether to overlap the current digit
        if random.random() < overlap_prob and current_x > 0:
            overlap_x = random.randint(int(new_width * 0.3), int(new_width * 0.7))
            current_x -= overlap_x  # Adjust x to create overlap

        if current_x + new_width > image_width:
            break

        composite_image.paste(digit_image, (current_x, y_offset), digit_image)

        # YOLO format: class_id, x_center, y_center, width, height
        x_center = normalize(current_x + new_width / 2, image_width)
        y_center = normalize(y_offset + new_height / 2, image_height)
        width = normalize(new_width, image_width)
        height = normalize(new_height, image_height)
        annotation_lines.append(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        current_x += new_width + horizontal_spacing

    # Save image
    image_path = os.path.join(images_dir, f"{idx}.jpg")
    composite_image.save(image_path)

    # Save annotations
    if annotation_lines:
        label_path = os.path.join(labels_dir, f"{idx}.txt")
        with open(label_path, "w") as label_file:
            label_file.write("\n".join(annotation_lines))
