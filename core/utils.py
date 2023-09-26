import os
from typing import List
import numpy as np
from PIL import Image, ImageFile

# config for PIL "IOError: image file truncated" error
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(image_path: str, colormode: str = "RGB") -> np.ndarray:
    """
    Reads an image from a file and returns a numpy array.
    """
    image = Image.open(image_path).convert(colormode)
    try:
        # If the image has EXIF data, check if it has orientation information.
        if hasattr(image, "_exif"):
            exifdata = image.getexif()
            if exifdata is not None:
                orientation = exifdata.get(274)
                if orientation == 3:
                    image = image.transpose(method=Image.ROTATE_180)
                elif orientation == 6:
                    image = image.transpose(method=Image.ROTATE_270)
                elif orientation == 8:
                    image = image.transpose(method=Image.ROTATE_90)
    except:
        # print(f"Could not read EXIF data from image {image_path}.")
        pass
    return np.array(image)


def get_files(path:str)-> List[str]:
    """
    get directory of all files in the folder.
    """
    allfiles = []
    for root,_,files in os.walk(path):
        for file in files:
            if not file[0]=='.':
                allfiles.append(os.path.join(root,file))
    return allfiles
