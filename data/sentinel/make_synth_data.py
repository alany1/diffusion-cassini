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
    original_data_root = Proto(env="DATASETS")
    original_data_prefix = "sentinel/original"

    data_root = Proto(env="DATASETS")
    data_prefix = "sentinel/noised/v0"

    samples_per_class = 1000
    cropped_size = 112

    L_min = 1
    L_max = 10

    
    p_clean = 0.05 # probably to use the clean version



def apply_gamma_noise(img: Image.Image, L=1):
    arr = np.array(img).astype(np.float32)
    noise = np.random.gamma(shape=L, scale=1.0 / L, size=arr.shape)
    noised_img = arr * noise 
    noised_img = np.clip(noised_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noised_img)

def save_img(img: Image.Image, out_fp):
    img.save(out_fp, format="PNG")

def crop_and_resize(fp, cropped_size):
    """
    """
    img = Image.open(fp)

    assert cropped_size <= img.size[0] and cropped_size <= img.size[1]
    img_gray = img.convert("L")
    

    w, h     = img_gray.size
    left     = (w - cropped_size) // 2
    top      = (h - cropped_size) // 2
    right    = left + cropped_size
    bottom   = top  + cropped_size
    box      = (left, top, right, bottom)

    img_crop = img_gray.crop(box)
    return img_crop

def entrypoint(**deps):
    DataArgs._update(**deps)

    all_image_paths = set()

    id = 0 # assign each outputted image an id so don't have duplicate filenames

    for terrain_type in ["agri", "barrenland", "grassland", "urban"]:
        # iterate thorugh each image in the folder
        directory_fp = f"{DataArgs.original_data_root}/{DataArgs.original_data_prefix}/{terrain_type}/s2"

        all_files = [
            directory_fp + "/" + fname
            for fname in os.listdir(directory_fp)
            if fname.lower().endswith(".png")
            and os.path.isfile(os.path.join(directory_fp, fname))
        ]


        print("samples_per_class", DataArgs.samples_per_class)
        print("len(all_files)", len(all_files) )


        assert DataArgs.samples_per_class <= len(all_files)

        all_files = all_files[:DataArgs.samples_per_class]

        print("len(all_files)", len(all_files) )

        Ls = np.random.randint(DataArgs.L_min, DataArgs.L_max + 1, size=DataArgs.samples_per_class)

        os.makedirs(f"{DataArgs.data_root}/{DataArgs.data_prefix}/{terrain_type}/", exist_ok=True)
        for L, file in zip(Ls, all_files):
            
            if random.random() < DataArgs.p_clean:
                # create a noise free image
                L = None
                img_crop = crop_and_resize(file, DataArgs.cropped_size) # TODO: add check that cropped size is leq than image size

                img_filename = f"{DataArgs.data_root}/{DataArgs.data_prefix}/{terrain_type}/terrain_{terrain_type}_L_{L}_id_{id}.png"
                save_img(img_crop, img_filename)
                all_image_paths.add(img_filename)
                id += 1
                
            else:
                # create a noised image
                img_crop = crop_and_resize(file, DataArgs.cropped_size)
                img_noise = apply_gamma_noise(img_crop, L=L)

                img_filename = f"{DataArgs.data_root}/{DataArgs.data_prefix}/{terrain_type}/terrain_{terrain_type}_L_{L}_id_{id}.png"
                save_img(img_noise, img_filename)
                all_image_paths.add(img_filename)
                id += 1
            
            



# Example usage:
if __name__ == "__main__":
    entrypoint()

    # debug
    # fp = rf"/Users/brianli/Desktop/diffusion_models/datasets/sentinel/original/agri/s2/ROIs1868_summer_s2_59_p2.png"
    # img_crop = crop_and_resize(fp, 112)

    # L = 90
    # img_type = "summer"
    # p=2

    # img_noise = apply_gamma_noise(img_crop, L)
    # tag = f"{img_type}_L_{L}_p{p}.png"
    # save_img(img_noise, rf"/Users/brianli/Desktop/diffusion_models/datasets/sentinel/test/{tag}")

