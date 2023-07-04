from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import numbers
import types
import cv2
from skimage.morphology import dilation

class JointResize(object):
    """Resize the input PIL.Image to the given 'size'.
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, img_size, resize_factor=1.0, interpolation=Image.BILINEAR):
        self.resize_factor = resize_factor
        self.img_size = img_size
        self.interpolation = interpolation

    def __call__(self, imgs):
        if self.resize_factor == 'dynamic':
            size = 1.0 + random.randint(0, 12) / 100.0
            size = int(size*self.img_size)
        else:
            size = int(self.resize_factor*self.img_size)
        out_list = []
        for i in range(len(imgs)):
            if i == 0:
                out_list.append(imgs[i].resize((size, size), self.interpolation))
            else:
                out_list.append(imgs[i].resize((size, size), Image.NEAREST))
        return out_list


class JointZoomOut(object):
    """Random zoom out the given list of PIL.Image to a random size of (0.5 to 1.0) of the original size
    """

    def __init__(self, zoom_factor=1.0, interpolation=Image.BILINEAR):
        self.zoom_factor = zoom_factor
        self.interpolation = interpolation

    def __call__(self, imgs):
        w, h = imgs[0].size
        if self.zoom_factor == 'dynamic':
            factor = random.uniform(0.5, 1.0)
            ow = int(w * factor)
            oh = int(h * factor)
        else:
            ow = int(w * self.zoom_factor)
            oh = int(h * self.zoom_factor)
        out_list = []
        for i in range(len(imgs)):
            if i == 0:
                out_list.append(imgs[i].resize((ow, oh), self.interpolation))
            else:
                out_list.append(imgs[i].resize((ow, oh), Image.NEAREST))
        return out_list


class JointScale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        w, h = imgs[0].size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return imgs
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            out_list = []
            for i in range(len(imgs)):
                if i == 0:
                    out_list.append(imgs[i].resize((ow, oh), self.interpolation))
                else:
                    out_list.append(imgs[i].resize((ow, oh), Image.NEAREST))
            return out_list
            # return [img.resize((ow, oh), self.interpolation) for img in imgs]
        else:
            oh = self.size
            ow = int(self.size * w / h)
            out_list = []
            for i in range(len(imgs)):
                if i == 0:
                    out_list.append(imgs[i].resize((ow, oh), self.interpolation))
                else:
                    out_list.append(imgs[i].resize((ow, oh), Image.NEAREST))
            return out_list


class JointCenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        w, h = imgs[0].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs]


class JointPad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, imgs):
        return [ImageOps.expand(img, border=self.padding, fill=self.fill) for img in imgs]


class JointLambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, imgs):
        return [self.lambd(img) for img in imgs]


class JointRandomCrop(object):
    """Crops the given list of PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, imgs):
        if self.padding > 0:
            imgs = [ImageOps.expand(img, border=self.padding, fill=0) for img in imgs]

        w, h = imgs[0].size
        th, tw = self.size
        if w == tw and h == th:
            return imgs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs]


class JointRandomHorizontalFlip(object):
    """Randomly horizontally flips the given list of PIL.Image with a probability of 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        return imgs

class JointRandomVerticalFlip(object):
    """Randomly vertically flips the given list of PIL.Image with a probability of 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            return [img.transpose(Image.FLIP_TOP_BOTTOM) for img in imgs]
        return imgs

class JointRandomRotation(object):
    """Randomly rotate the given list of PIL.Image
    """

    def __init__(self, degree=45., expend=True, p=None):
        self.degree = degree
        self.expend = expend
        if p is not None:
            self.p = p
        else:
            self.p = 1.0

    def __call__(self, imgs):
        if random.random() < self.p:
            if self.degree == 'dynamic':
                d = random.uniform(15., 45.)
                return [img.rotate(d, expand=self.expend) for img in imgs]
            else:
                d = self.degree
                return [img.rotate(d, expand=self.expend) for img in imgs]
        return imgs


class JointRandomSizedCrop(object):
    """Random crop the given list of PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        for attempt in range(10):
            area = imgs[0].size[0] * imgs[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= imgs[0].size[0] and h <= imgs[0].size[1]:
                x1 = random.randint(0, imgs[0].size[0] - w)
                y1 = random.randint(0, imgs[0].size[1] - h)

                imgs = [img.crop((x1, y1, x1 + w, y1 + h)) for img in imgs]
                assert(imgs[0].size == (w, h))

                out_list = []
                for i in range(len(imgs)):
                    if i == 0:
                        out_list.append(imgs[i].resize((self.size, self.size), self.interpolation))
                    else:
                        out_list.append(imgs[i].resize((self.size, self.size), Image.NEAREST))
                return out_list

        # Fallback
        scale = JointScale(self.size, interpolation=self.interpolation)
        crop = JointCenterCrop(self.size)
        return crop(scale(imgs))

class JointRandomObjectCrop(object):
    """Random crop the given list of PIL.Image to a random size of (from object size to 1.0) of the original size
    """

    def __init__(self, padding=0):
        self.padding = padding

    def __call__(self, imgs):

        w, h = imgs[0].size
        x, y, tw, th = self.get_boundingRect(imgs[1])
        if w == tw and h == th:
            return imgs

        x1 = random.randint(0, x)
        y1 = random.randint(0, y)

        x2 = random.randint(x + tw, w)
        y2 = random.randint(y + th, h)
        return [img.crop((x1, y1, x2, y2)) for img in imgs]

    def get_boundingRect(self, mask):
        m = np.asarray(mask)

        contours, hie = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        inline = []
        for i in range(len(contours)):
            if hie[0][i][-1] == -1 and cv2.contourArea(contours[i]) > 10:
                inline.append(i)
        re = np.vstack([contours[i] for i in inline])
        x, y, w, h = cv2.boundingRect(re)
        return x, y, w, h

class JointMaskDilation(object):
    """Dilate the mask PIL.Image in the given lists, with 15 pixels;
       if not dilate with 15 pixels, change the hype-params,
    """

    def __init__(self, img_heigh=31, img_width=31, radius=16, center_x=15, center_y=15, padding=0):
        self.img_heigh = img_heigh
        self.img_width = img_width
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y

        self.padding = padding

    def __call__(self, imgs):
        if self.padding > 0:
            imgs = [ImageOps.expand(img, border=self.padding, fill=0) for img in imgs]

        w, h = imgs[0].size
        ker = self.cir_kernel(self.img_heigh, self.img_width, self.radius, self.center_x, self.center_y)
        dilated_mask = Image.fromarray(dilation(imgs[1], ker))

        return [imgs[0], dilated_mask]

    def cir_kernel(self, img_heigh, img_width, radius, center_x, center_y):

        y, x = np.ogrid[0:img_heigh, 0:img_width]

        ker = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2).astype('uint8')

        return ker