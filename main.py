import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocess_data
from model import init_model, get_rays, render_rays

def main():
    train_images, train_poses, test_image, test_pose, focal = preprocess_data()
    H, W = train_images.shape[1:3]

    model = init_model()
    optimizer = tf.keras.optimizers.Adam(5e-4)

    N_samples = 64
    N_iters = 1000
    psnrs = []
    iternums = []
    i_plot = 25

    import time
    t = time.time()
    for i in range(N_iters+1):
        
        img_i = np.random.randint(train_images.shape[0])
        target = train_images[img_i]
        pose = train_poses[img_i]
        rays_o, rays_d = get_rays(H, W, focal, pose)
        with tf.GradientTape() as tape:
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
            loss = tf.reduce_mean(tf.square(rgb - target))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if i%i_plot==0:
            print(i, (time.time() - t) / i_plot, 'secs per iter')
            t = time.time()
            
            # Render the holdout view for logging
            rays_o, rays_d = get_rays(H, W, focal, test_pose)
            rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
            loss = tf.reduce_mean(tf.square(rgb - test_image))
            psnr = -10. * tf.math.log(loss) / tf.math.log(10.)

            psnrs.append(psnr.numpy())
            iternums.append(i)
            
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            plt.imshow(rgb)
            plt.title(f'Iteration: {i}')
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title('PSNR')
            plt.show()

    print('Done')

if __name__ == "__main__":
    main()