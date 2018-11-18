# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K


def read_img(img_path, preprocess_input, size):
    """util function to read and preprocess the test image.

    Args:
           img_path: path of image.
           preprocess_input: preprocess_input function.
           size: resize.

    Returns:
           img: original image.
           pimg: processed image.
    """
    img = cv2.imread(img_path)
    pimg = cv2.resize(img, size)

    pimg = np.expand_dims(pimg, axis=0)
    pimg = preprocess_input(pimg)

    return img, pimg


def deprocess_image(x):
    """util function to convert a tensor into a valid image.

    Args:
           x: tensor of filter.

    Returns:
           x: deprocessed tensor.
    """
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def normalize(x):
    """utility function to normalize a tensor by its L2 norm

    Args:
           x: gradient.

    Returns:
           x: gradient.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def vis_conv(images, n, name, t):
    """visualize conv output and conv filter.

    Args:
           img: original image.
           n: number of col and row.
           t: vis type.
           name: save name.
    """
    size = 64
    margin = 5

    if t == 'filter':
        results = np.zeros((n * size + 7 * margin, n * size + 7 * margin, 3))
    if t == 'conv':
        results = np.zeros((n * size + 7 * margin, n * size + 7 * margin))

    for i in range(n):
        for j in range(n):
            if t == 'filter':
                filter_img = images[i + (j * n)]
            if t == 'conv':
                filter_img = images[..., i + (j * n)]
            filter_img = cv2.resize(filter_img, (size, size))

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            if t == 'filter':
                results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
            if t == 'conv':
                results[horizontal_start: horizontal_end, vertical_start: vertical_end] = filter_img

    # Display the results grid
    plt.imshow(results)
    plt.savefig('images\{}_{}.jpg'.format(t, name), dpi=600)
    plt.show()


def vis_heatmap(img, heatmap):
    """visualize heatmap.

    Args:
           img: original image.
           heatmap：heatmap.
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    plt.figure()

    plt.subplot(221)
    plt.imshow(cv2.resize(img, (224, 224)))
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(heatmap)
    plt.axis('off')

    plt.subplot(212)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    # We apply the heatmap to the original image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    plt.imshow(superimposed_img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('images\heatmap.jpg', dpi=600)
    plt.show()
