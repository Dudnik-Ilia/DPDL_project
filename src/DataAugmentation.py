import numpy as np

from torchvision import transforms, utils
import torch

class ToPILImage(object):
  def __init__(self):
    self.transform = transforms.ToPILImage()
  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']
              
    return {'image': self.transform(image), 'landmarks': landmarks}


class Normalize(object):
    """Normalize the image intensities in an image.
    *Note*: This is a somewhat tricky part, i.e., on which basis to normalize 
    (min-max normalization, zero-mean/unit-variance), whether to normalize 
    per-image or over the whole data set, before/after data augmentation, etc.

    Additionally, some convenience methods also perform some implicit 
    normalization, e.g., scaling the data between 0-1. Make sure that you check 
    the statistics of your training, validation and test data repeatedly!

    In this function, we will go for per-sample min-max normalization out of 
    convenience. Zero-mean-unit-variance is typically more numerically stable.
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        img_max = torch.max(image)
        img_min = torch.min(image)

        img = (image - img_min)/(img_max - img_min)
        
        return {'image': img, 'landmarks': landmarks}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        w, h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        trnsf = transforms.Resize((new_w, new_h))

        img = trnsf(image)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, max_crop_factor = 0.9, prob = 0.5):
        self.prob = prob
        self.max_crop = max_crop_factor

    def __call__(self, sample):
        p = torch.rand(1)
        if p > self.prob:
          return sample 

        image, landmarks = sample['image'], sample['landmarks']

        w, h = image.size
        h, w = int(h), int(w)
        v = (self.max_crop + torch.randn(1) % (1-self.max_crop))
        new_h, new_w = h * v, w * v
        new_h, new_w = int(new_h.item()), int(new_w.item())
        top = int(torch.randint(0, h - new_h, [1]))
        left = int(torch.randint(0, w - new_w, [1]))

        image = image.crop((left, top, left + new_w, top + new_h))

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ColorJitter(object):

  def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
    self.transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']

    image = self.transform(image)
    return {'image': image,  ## hacky! Do this better in your code by arranging the augmentation in the right order!
            'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
      self.transform = transforms.ToTensor()
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1)).astype('float64')
        landmarks /= np.array(image.size, dtype=float)
        landmarks = landmarks.reshape(4)
        return {'image': self.transform(image),
                'landmarks': torch.from_numpy(landmarks)}