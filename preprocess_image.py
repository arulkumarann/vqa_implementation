import os
import argparse
from PIL import Image

def resize_image(image, size):
    """ resizes the given image to the speccified size"""
    return image.resize(size, Image.Resampling.LANCZOS)

def resize_image_in_directory(input_path, output_path, size):
    """resizes image from 'input_path' and saves onto 'output_path'"""
    img_size = size
    for idir in os.scandir(input_path):
        if not idir.is_dir():
            continue
        if not os.path.exists(output_path+ "/"+ idir.name):
            os.makedirs(output_path + "/" + idir.name)
        images = os.listdir(idir.path)
        num_images = len(images)
        for iimage, image in enumerate(images):
            try:
                with open(os.path.join(idir.path, image), "r+b") as f:
                    with Image.open(f) as img:
                        img = resize_image(img,img_size)
                        img.save(os.path.join(output_path + "/" + idir.name, image), img.format)
            except (IOError, SyntaxError) as e:
                pass
            if (iimage+1) % 1000 == 0:
                print("[{}/{}] resized imgs and saved them to '{}'". format(iimage+1, num_images, output_path + '/' + idir.name))

input_path = r"/teamspace/studios/this_studio/vqa_implementation/datasets/Images"
output_path = r"/teamspace/studios/this_studio/vqa_implementation/datasets/Resized_Images"
img_size = (224,224)

resize_image_in_directory(input_path, output_path,size = img_size)