# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

ssim = sk.metrics.structural_similarity

# name of the input file
imname = 'data/emir.tif'

# read in the image
im = skio.imread(imname)

# Convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(np.int32)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def align(img1, img2, shift = 30):
    row_shift, col_shift = 0, 0
    dist = 0
    for i in range(-shift, shift):
        if i < 0:
            r_shift1 = img1[-i:]
            r_shift2 = img2[:i]
        elif i == 0:
            r_shift1 = img1
            r_shift2 = img2
        else:
            r_shift1 = img1[:-i]
            r_shift2 = img2[i:]
        for j in range(-shift, shift):
            if j < 0:
                c_shift1 = r_shift1[:, -j:]
                c_shift2 = r_shift2[:, :j]
            elif j == 0:
                c_shift1 = r_shift1
                c_shift2 = r_shift2
            else:
                c_shift1 = r_shift1[:, :-j]
                c_shift2 = r_shift2[:, j:]
            # img1_shifted = np.roll(img1, (i, j), axis = (0, 1))
            
            sim = ssim(c_shift1, c_shift2, data_range = 1)
            if sim > dist:
                dist = sim
                row_shift = i
                col_shift = j
    return row_shift, col_shift

def pyramid_align(img1, img2):
    h, w = img1.shape
    exp = int(np.log2(min(h, w) / 100))
    factor = pow(2, max(exp, 0))
    simg1 = sk.transform.downscale_local_mean(img1, (factor, factor))
    simg2 = sk.transform.downscale_local_mean(img2, (factor, factor))

    r, c = align(simg1, simg2, 20)

    while exp > 0:
        exp -= 1
        factor = pow(2, exp)
        simg1 = sk.transform.downscale_local_mean(img1, (factor, factor))
        simg2 = sk.transform.downscale_local_mean(img2, (factor, factor))
        prev_r, prev_c = (2 * r, 2 * c)
        simg1_shift = np.roll(simg1, (prev_r, prev_c), axis = (0, 1))
        new_r, new_c = align(simg1_shift, simg2, 2)
        r, c = prev_r + new_r, prev_c + new_c

    return r, c

ag = pyramid_align(g, b)
ar = pyramid_align(r, b)

print(ag, ar)

shifted_g = np.roll(g, ag, axis = (0, 1))
shifted_r = np.roll(r, ar, axis = (0, 1))
# create a color image
im_out = np.dstack([shifted_r, shifted_g, b])

im_out = sk.exposure.equalize_adapthist(im_out)

im_out_uint = (255 * im_out).astype(np.uint8)

# save the image
fname = 'new images/emir_contrast.jpg'
skio.imsave(fname, im_out_uint)

# display the image
skio.imshow(im_out)
skio.show()