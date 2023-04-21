import openslide
from pathlib import Path
import numpy as np
from random import randint
from torch.utils.data import Dataset
import torch

from typing import Union
import viz_utils


class SlideContainer:

    def __init__(self, file: Union[Path, str], image_id: int, y, level: int = 0, width: int = 256, height: int = 256, sample_func: callable = None):
        self.file = file
        self.image_id = image_id
        self.slide = openslide.open_slide(str(file))
        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]
        self.targets = y
        # self.sample_func = sample_func
        self.classes = list(set(self.targets[1]))

        if level is None:
            level = self.slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int = 0, y: int = 0):
        """
        Cuts the image at x,y coords
        """
        try:
            arr = np.copy(np.array(self.slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=self.level, size=(self.width, self.height)))[:, :, :3])
            # print(f"Read image gracefully {self.file}")
            return arr

        except OSError as error:
            width, height = self.slide_shape
            # print(f"{error} for region ({x}, {y}, {self.width}, {self.height}) for image with dimensions ({width}, {height}) \n{self.file})")
            return np.zeros((self.width, self.height, 3), dtype=np.uint8)

    @property
    def shape(self):
        return self.width, self.height

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self.level]

    def __str__(self):
        return str(self.file)

    def get_new_train_coordinates(self, sample_func=None):
        """
        Outputs: x0, y0 of a patch
        The standard version cuts the picture to the size of the patch
        on the bottom and right and selects a stupidly random point.
        """
        # TODO: Implement a sampling method that can be provided via sample_func
        patch_dimensions = (self.width, self.height)
        # image dims
        width, height = self.slide.level_dimensions[self.level]
        # use passed sampling method
        if callable(sample_func):
            """return sample_func(self.targets, **{"classes": self.classes, "shape": self.shape,
                                                     "level_dimensions": self.slide.level_dimensions,
                                                     "level": self.level})
            """
            return sample_func(self.targets,
                               (width, height),
                               patch_dimensions)

        # use default sampling method
        # here is the cut that assures of picking patch that is not beyond size of the image
        x0,y0 = np.random.randint(0, width - self.shape[0]), np.random.randint(0, height - self.shape[1])
        return x0,y0


def custom_sampling_fun(targets, dimensions, patch_dims):
    """
    Inputs:
        targets: boxes and labels
        dimensions: width and height of the whole image
        patch_dims: size of the patch
    """
    bboxes, labels = targets
    width, height = dimensions

    # pick random annotation on the image
    i = np.random.randint(len(bboxes))
    x_0_ROI, y_0_ROI, x_1_ROI, y_1_ROI = bboxes[i]
    # find an anchor (center)
    anchor_x, anchor_y = (x_1_ROI+x_0_ROI)/2, (y_1_ROI+y_0_ROI)/2
    # pick the shift of the patch relative to anchor
    # be sure that annotation is inside:

    if (x_1_ROI-x_0_ROI) < patch_dims[0] or (y_1_ROI-y_0_ROI) < patch_dims[1]:
        raise ValueError("The annotation box is bigger than the patch size!")

    # minimum is the half of the annotation box (because with respect to the center(anchor))
    # maximum is the patch size minus half annotation box size
    anchor_shift_x = np.random.randint((x_1_ROI-x_0_ROI)/2,
                                       patch_dims[0]-((x_1_ROI-x_0_ROI)/2))
    anchor_shift_y = np.random.randint((y_1_ROI-y_0_ROI)/2,
                                       patch_dims[1]-((y_1_ROI-y_0_ROI)/2))
    x0 = anchor_x - anchor_shift_x
    y0 = anchor_y - anchor_shift_y
    # check if we are beyond
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x0+patch_dims[0] >= width:
        x0 = width - patch_dims[0]
    if y0+patch_dims[1] >= height:
        y0 = height - patch_dims[1]

    return x0, y0


class MIDOGTrainDataset(Dataset):

    def __init__(self, list_containers: list[SlideContainer], patches_per_slide_container: int = 10, transform=None, sample_func=None) -> None:
        super().__init__()
        self.list_containers = list_containers
        # note that we are working with pseudo-epochs. We could also exact a set amount of patches per 
        # image region but sampling them randomly typically adds additional variability to the samples
        self.patches_per_slide_container = patches_per_slide_container
        self.transform = transform
        self.sample_func = sample_func

    def __len__(self):
        return len(self.list_containers)*self.patches_per_slide_container

    def get_patch_w_labels(self, cur_container: SlideContainer, x: int=0, y: int=0):
        """
        inputs: x,y coords of training sample
        returns patch (tensor) and target (what form?)
        """

        # cut the image 512 size we set up (256 default)
        patch = cur_container.get_patch(x, y)
    
        bboxes, labels = cur_container.targets
        # cut size - 512
        h, w = cur_container.shape

        # they are being changed to np but they already are
        bboxes = np.array([box for box in bboxes]) if len(np.array(bboxes).shape) == 1 else np.array(bboxes)
        labels = np.array(labels)

        area = np.empty([0])

        # why to check, if it is train and labels are there?
        if len(labels) > 0:
            # Bboxes adjustment for the cut image
            bboxes, labels = viz_utils.filter_bboxes(bboxes, labels, x, y, w, h)

            # not ideal, but ensuring consistency (not cutting bboxes before augmentation) is a massive amount of work with likely fairly little additional benefit
            if self.transform:
                transformed = self.transform(image=patch, bboxes=bboxes, class_labels=labels)
                patch = transformed['image']
                bboxes = np.array(transformed['bboxes'])
                labels = transformed['class_labels']

            if len(bboxes) > 0:
                area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        else:
            if self.transform:
                patch = self.transform(image=patch, bboxes=bboxes, class_labels=labels)['image']
        
        bboxes = bboxes.reshape((-1, 4))

        # following the label definition described here: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#torchvision-object-detection-finetuning-tutorial
        targets = {
            'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([cur_container.image_id]),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
            'area': torch.as_tensor(area, dtype=torch.float32)
        }

        patch_as_tensor = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float()/255.0
        return patch_as_tensor, targets

    def __getitem__(self, idx):
        # select the container we want to sample from
        idx_slide = idx % len(self.list_containers)
        cur_image_container = self.list_containers[idx_slide]
        # by default just random cut from the image
        train_coordinates = cur_image_container.get_new_train_coordinates(self.sample_func)

        return self.get_patch_w_labels(cur_image_container, *train_coordinates)
    

class MIDOGTestDataset(Dataset):

    def __init__(self, container: SlideContainer, nms_threshold=0.4):
        self.nms_threshold = nms_threshold

        self.container = container
        
        # TODO: Add additional member variables

    def __len__(self) -> int:
        # TODO: replace the following line: return number of patches from this container
        return 0

    def __getitem__(self, idx):
        patch_width, patch_height = self.container.width, self.container.height

        # TODO: replace the following lines: get patch for this idx
        patch = np.zeros((patch_width, patch_height))

        # TODO: replace the following lines: get corresponding targets for the patch
        bboxes = []
        labels = []
        area = []
        
        # transforming it to pytorch tensor
        patch_as_tensor = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float() / 255.0
        targets = {
            'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([self.container.image_id]),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
            'area': torch.as_tensor(area, dtype=torch.float32)
        }

        return patch_as_tensor, targets

    def get_slide_labels_as_dict(self) -> dict:
        bboxes, labels = self.container.targets
        bboxes = bboxes.reshape((-1, 4))
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        targets = {
            'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([self.container.image_id]),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
            'area': torch.as_tensor(area, dtype=torch.float32)
        }

        return targets

    def local_to_global(self, idx, bboxes:torch.Tensor) -> torch.Tensor:
        # TODO: replace the following lines: Transform local patch / bbox coordinates to global (RoI-wise ones)
        bboxes_global = []

        return bboxes_global
