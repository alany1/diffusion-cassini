from params_proto import ParamsProto, Proto
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import trange
import os
import random
from matplotlib import pyplot as plt


class DataArgs(ParamsProto):
    """
    Create MNIST-like synthetic data, following the noise model.
    """
    data_root = Proto(env="DATASETS")
    data_prefix = "mnist/v0"

    L_min = 1
    L_max = 64

    digit_min = 0
    digit_max = 9

    bg_color_min = 0.1
    bg_color_max = 0.3

    samples_per_digit = 25

    image_size = (28, 28)
    font_path = "DejaVuSans-Bold.ttf"
    font_size = 20


def create_digit_base(digit, image_size, font_path, font_size, bg_color):
    bg_val = int(np.clip(bg_color, 0, 1) * 255)

    img = Image.new("L", image_size, color=bg_val)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = (
        int((image_size[0] - text_width) / 2 - bbox[0]),
        int((image_size[1] - text_height) / 2 - bbox[1])
    )

    draw.text(position, text, fill=255, font=font)

    img_array = np.array(img)
    base_img = np.where(img_array > bg_val, 255, bg_val).astype(np.float32)

    return base_img


def create_noised_digit(digit, L=1, image_size=(28, 28), font_path="DejaVuSans-Bold.ttf", font_size=20, bg_color=0.5):
    base_img = create_digit_base(digit, image_size, font_path, font_size, bg_color)
    noise = np.random.gamma(shape=L, scale=1.0 / L, size=base_img.shape)
    noised_img = base_img * noise

    noised_img = np.clip(noised_img, 0, 255).astype(np.uint8)
    return noised_img


def create_ground_truth_digit(digit, image_size=(28, 28), font_path="DejaVuSans-Bold.ttf", font_size=20, bg_color=0.5):
    ground_truth = create_digit_base(digit, image_size, font_path, font_size, bg_color)
    return ground_truth.astype(np.uint8)


def entrypoint(**deps):
    DataArgs._update(**deps)

    all_image_paths = set()

    for digit in trange(DataArgs.digit_min, DataArgs.digit_max + 1, desc="Generating digits!"):
        Ls = np.random.randint(DataArgs.L_min, DataArgs.L_max + 1, size=DataArgs.samples_per_digit)
        bg_colors = np.random.uniform(DataArgs.bg_color_min, DataArgs.bg_color_max, size=DataArgs.samples_per_digit)

        os.makedirs(f"{DataArgs.data_root}/{DataArgs.data_prefix}/{digit}", exist_ok=True)

        for L, bg_color in zip(Ls, bg_colors):
            noised_image = create_noised_digit(
                digit,
                L=L,
                bg_color=bg_color,
                image_size=DataArgs.image_size,
                font_path=DataArgs.font_path,
                font_size=DataArgs.font_size
            )

            img_filename = f"{DataArgs.data_root}/{DataArgs.data_prefix}/{digit}/digit_{digit}_L_{L}_bg_{bg_color:.2f}.png"
            Image.fromarray(noised_image).save(img_filename)

            all_image_paths.add(img_filename)

    sample_records = random.sample(list(all_image_paths), min(10, len(all_image_paths)))

    num_samples = len(sample_records)
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i, img_path in enumerate(sample_records):
        noised_img = np.array(Image.open(img_path))
        axes[0, i].imshow(noised_img, cmap="gray")
        axes[0, i].set_title(os.path.basename(img_path), fontsize=8)
        axes[0, i].axis("off")

        base = os.path.basename(img_path)
        parts = base.split("_")
        digit = int(parts[1])
        bg_str = parts[5].split(".")[0]  # Extract bg color as string
        bg_color = float(bg_str)

        ground_truth = create_ground_truth_digit(
            digit,
            image_size=DataArgs.image_size,
            font_path=DataArgs.font_path,
            font_size=DataArgs.font_size,
            bg_color=bg_color
        )

        axes[1, i].imshow(ground_truth, cmap="gray")
        axes[1, i].set_title("Ground Truth", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    entrypoint()
