import numpy as np
import torch
import torchvision.transforms.functional
import image_sampler
import tqdm
import matplotlib.pyplot as plt

height = 640
width = 640
dummy_img = np.ones((1, height, width), dtype=np.float32)
dummy_img[:, :, width // 2 - 5:width // 2 + 5] = 0.0
dummy_img_torch = torch.from_numpy(dummy_img)

testing = False
if testing:
    for k in tqdm.tqdm(range(1000)):
        aug_img = image_sampler.randomly_augment_image(dummy_img_torch)
        assert aug_img.shape[1] == 640 and aug_img.shape[2] == 640
else:
    # plot image with matplotlib
    aug_img = image_sampler.randomly_augment_image(dummy_img_torch)
    grayscale_img = aug_img[0, ...]
    plt.imshow(grayscale_img, cmap="gray")
    plt.show()
