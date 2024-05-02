import zipfile
import numpy as np
from PIL import Image
import io
import os
from matplotlib import cm
import json

def process_zipfile():
    zip_file_path = './data.zip'
    extracted_directory = './extracted_files2/'

    # Open the zip file
    if not os.path.exists(extracted_directory):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all files from the zip
            zip_ref.extractall('extracted_files2')
    else:
        print("Directory already exists. Skipping extraction.")


def process_dict(nerf_obj, focal=138):
    # dictionary = dict(np.load('./data/tiny_nerf_data.npz'))
    # print(dictionary['images'].shape)
    # print(dictionary['poses'].shape)
    # print(dictionary['focal'].shape)
    extracted_directory = './extracted_files2/'
    images = []
    poses = []
    for root, dirs, files in os.walk(extracted_directory + f'data/{nerf_obj}/'):
        for file in files:
            if file.endswith('.png'):
                with open(os.path.join(root, file), 'rb') as f:
                    img = Image.open(io.BytesIO(f.read()))
                    img = np.array(img)
                    img = np.resize(img, (100, 100, 3))
                    images.append(img)
            elif file.endswith('.json'):
                with open(os.path.join(root,file), 'rb') as json_file:
                    data = json.load(json_file)['frames']
                    for d in data:
                        pose_matrix = np.array(d['transform_matrix'])
                        poses.append(pose_matrix)
                
    poses = np.array(poses)
    images = np.array(images)
    focal = np.array(focal)

    dictionary = {
        'poses' : poses,
        'images' : images,
        'focal' : focal
    }

    np.savez(f'./{nerf_obj}.npz', **dictionary)




def main():
    print("==================Preprocessing==================")
    process_zipfile()
    process_dict('bouncingballs')

if __name__ == "__main__":
    main()