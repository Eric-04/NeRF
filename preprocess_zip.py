import zipfile
import numpy as np
from PIL import Image
import io
import os
import json

extracted_directory_name = 'extracted_files'
extracted_directory = f'./{extracted_directory_name}/'
zip_file_path = './data.zip'

def process_zipfile():
    # Open the zip file
    if not os.path.exists(extracted_directory):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all files from the zip
            zip_ref.extractall('extracted_files')
    else:
        print("Directory already exists. Skipping extraction.")

def process_npzfile(nerf_obj, focal=138):
    # if os.path.exists(f'./data/{nerf_obj}.npz'):
    #     return
    images = []
    poses = []
    for root, dirs, files in os.walk(extracted_directory + f'data/{nerf_obj}/'):
        if files[0].endswith('.png'): files = sorted(files) # need to make sure that the images are in order
        for file in files:
            if file.endswith('.png'):
                with open(os.path.join(root, file), 'rb') as f:
                    img = Image.open(io.BytesIO(f.read()))
                    img = img.convert('RGB')
                    img = img.resize((100, 100), Image.LANCZOS)
                    img = np.array(img)
                    img = np.array(img).astype(np.float32) / 255.0
                    images.append(img)
            elif file.endswith('.json'):
                with open(os.path.join(root,file), 'rb') as json_file:
                    data = json.load(json_file)['frames']
                    for d in data:
                        pose_matrix = np.array(d['transform_matrix']).astype(np.float32)
                        poses.append(pose_matrix)
                
    poses = np.array(poses)
    images = np.array(images)
    focal = np.array(focal)

    dictionary = {
        'poses' : poses,
        'images' : images,
        'focal' : focal
    }

    os.makedirs('./data/', exist_ok=True)
    np.savez(f'./data/{nerf_obj}.npz', **dictionary)

def main():
    print("==================Preprocessing==================")
    process_zipfile()
    process_npzfile('bouncingballs')

if __name__ == "__main__":
    main()