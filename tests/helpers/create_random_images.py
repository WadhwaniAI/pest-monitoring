import os
import random

import numpy as np
from PIL import Image

np.random.seed(42)


def generate_random_images(num_images: int, path: str):
    sizes = []
    for i in range(num_images):
        height, width = random.randint(50, 150), random.randint(100, 200)
        sizes.append((height, width))
        img = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(path, f"{i}.jpeg"))
    return sizes


# Generate Images
save_path = "tests/helpers/resources/images/"
sizes = generate_random_images(200, save_path)
