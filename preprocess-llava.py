import os
import argparse
from PIL import Image

def resize_image_inplace(image_path, size=(512, 512)):
    image_a = Image.open(image_path).resize((512, 512))
    image_a = image_a.convert('RGB')
    image_a.save(image_path)

def process_directory(directory, size=(512, 512)):
    # print("aba")
    if not os.path.isdir(directory):
        print(f"{directory} is not a valid directory")
        return
    
    for filename in os.listdir(directory):
        print(directory)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            image_path = os.path.join(directory, filename)
            resize_image_inplace(image_path, size)

def concat_images(img_fol, gap_size):
    path_a = f'{img_fol}/0_0.png'
    path_b = f'{img_fol}/0_1.png'
    p = 512
    image_a = Image.open(path_a).resize((p, p))
    image_b = Image.open(path_b).resize((p, p))

    # Calculate new dimensions including gaps
    new_width = 2 * p +  gap_size  # width of a + gap + b + gap + c
    new_height = p  

    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    # Paste the images into the new image with gaps
    new_image.paste(image_a, (0, 0))
    new_image.paste(image_b, (p + gap_size, 0))
    print("concat path: ", f'{img_fol}/concat.png')
    new_image.save(f'{img_fol}/concat.png')
    print(f"Output saved to {img_fol}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in a directory to 512x512 pixels")
    parser.add_argument("--directory", type=str, help="Path to the directory containing images")
    
    args = parser.parse_args()
    root_dir = args.directory
    subdirs= []
    for root, dirs, files in os.walk(root_dir):
        for subdir in dirs:
            if ('_' in subdir and 'ipynb' not in subdir):
                process_directory(os.path.join(root_dir, subdir))
                concat_images(os.path.join(root_dir, subdir), 10)
# python3 preprocess-llava.py --directory [Path to image_folder]
