import os
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
from preprocess_zip import process_zipfile, process_npzfile

def preprocess_data():

    process_zipfile()

    nerf_objects = ['bouncingballs', 'hellwarrior', 'hook', 'jumpingjacks', 'lego', 'mutant', 'standup', 'trex']
    for obj in nerf_objects:
        process_npzfile(obj)

    # if not os.path.exists('./bouncingballs.npz'):
    #     raise RuntimeError("Data file does not exist. Please download the data using bash download_example_data.sh")

    data = np.load('./data/bouncingballs.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1:3]
    print(images.shape, poses.shape, focal)

    test_image, test_pose = images[101], poses[101]
    train_images, train_poses = images[:100,...,:3], poses[:100]

    plt.imshow(test_image)
    plt.show()

    return train_images, train_poses, test_image, test_pose, focal