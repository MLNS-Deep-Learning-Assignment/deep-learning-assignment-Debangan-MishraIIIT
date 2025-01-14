import random
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageDraw

class MultiMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_data, image_size=(40, 168), prob_4_digits=0.9):
        self.mnist_data = mnist_data
        self.image_height, self.image_width = image_size
        self.prob_4_digits = prob_4_digits

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        num_digits = random.choices([3, 4, 5], weights=[0.05, 0.9, 0.05])[0]
        digits = random.choices(range(len(self.mnist_data)), k=num_digits)

        composite_image = Image.new("L", (self.image_width, self.image_height), color=0)
        target = ""

        for i, digit_idx in enumerate(digits):
            digit_image, label = self.mnist_data[digit_idx]
            target += str(label)

            scale = random.uniform(0.8, 1.2)
            new_width = int(digit_image.width * scale)
            new_height = int(digit_image.height * scale)
            digit_image = digit_image.resize((new_width, new_height))

            # Randomly place the digit in the composite image
            max_x = self.image_width - new_width
            max_y = self.image_height - new_height

            x_offset = random.randint(0, max_x)
            y_offset = random.randint(0, max_y)

            composite_image.paste(digit_image, (x_offset, y_offset))

        # Convert image to tensor
        transform = transforms.ToTensor()
        composite_image = transform(composite_image)

        return composite_image, target

# Usage example
# Usage example
if __name__ == "__main__":
    # Load MNIST dataset with proper transformations

    mnist_train = datasets.MNIST(root="./data", train=True, download=True)

    # Create MultiMNIST dataset
    multi_mnist = MultiMNISTDataset(mnist_train, image_size=(40, 168))

    # Example: Get a sample and visualize
    sample_image, sample_target = multi_mnist[0]

    print(f"Target: {sample_target}")
    transforms.ToPILImage()(sample_image).show()
