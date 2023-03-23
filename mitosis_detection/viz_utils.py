from dpdl_defaults import *
import openslide
import os.path
import numpy as np
import torch


def tlbr2cthw(boxes):
    """Convert top/left bottom/right format `boxes` to center/size corners."""
    center = (boxes[:, :2] + boxes[:, 2:])/2
    sizes = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([center, sizes], 1)


def cthw2tlbr(boxes):
    """Convert center/size format `boxes` to top/left bottom/right corners."""
    top_left = boxes[:, :2] - boxes[:, 2:]/2
    bot_right = boxes[:, :2] + boxes[:, 2:]/2
    return torch.cat([top_left, bot_right], 1)


def rescale_box(bboxes, size: torch.Tensor):
    bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:] / 2
    bboxes[:, :2] = (bboxes[:, :2] + 1) * size / 2
    bboxes[:, 2:] = bboxes[:, 2:] * size / 2
    bboxes = bboxes.long()
    return bboxes


def tlbr2cthw(boxes):
    """Convert top/left bottom/right format `boxes` to center/size corners."""
    center = (boxes[:, :2] + boxes[:, 2:])/2
    sizes = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([center, sizes], 1)


def encode_class(idxs, n_classes):
    target = idxs.new_zeros(len(idxs), n_classes).float()
    mask = idxs != 0
    i1s = torch.LongTensor(list(range(len(idxs))))
    target[i1s[mask], idxs[mask]-1] = 1
    return target


def cthw2tlbr(boxes):
    """Convert center/size format `boxes` to top/left bottom/right corners."""
    top_left = boxes[:, :2] - boxes[:, 2:]/2
    bot_right = boxes[:, :2] + boxes[:, 2:]/2
    return torch.cat([top_left, bot_right], 1)


def collate_fn(batch):
    return tuple(zip(*batch))


def image_filename2id(image_filename):
    return int(os.path.splitext(image_filename)[0])


def image_id2filename(image_id):
    return "{image_id:03f}".format() + ".tiff"


def get_colors_for_labels(labels, dict_colors={1: "yellow", 2: "blue"}):
    return [dict_colors[l] for l in labels]


def filter_bboxes(bboxes, labels, x0, y0, w, h):
    # generate copy of boxes
    if len(bboxes) != len(labels):
         raise ValueError("lists bboxes and classes should have the same length but have length {} and {}".format(len(bboxes), len(labels)))

    if len(labels) > 0:
        bboxes = np.copy(bboxes)

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x0
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y0

        bb_half_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2  # if half of the box is still contained in the image -- keep
        bb_half_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2
        ids = ((bboxes[:, 0] + bb_half_widths) > 0) \
                & ((bboxes[:, 1] + bb_half_heights) > 0) \
                & ((bboxes[:, 2] - bb_half_widths) < w) \
                & ((bboxes[:, 3] - bb_half_heights) < h)

        bboxes = bboxes[ids]
        labels = labels[ids]
        bboxes = np.clip(bboxes, 0, max(h, w))  # not an issue for quadratic regions, but wouldn't this be problematic otherwise?

    return bboxes, labels


def get_bboxes(image_filename: str, annotation_data, categories: list=[1, 2]):
    image_id = image_filename2id(image_filename)
    annotations = [anno for anno in annotation_data['annotations'] if anno["image_id"] == image_id and anno["category_id"] in categories]
    bboxes = [a["bbox"] for a in annotations]
    labels = [a["category_id"] for a in annotations]

    bboxes = np.array(bboxes).reshape((-1, 4))
    labels = np.array(labels)

    return bboxes, labels


def get_image_and_bboxes(image_filename, annotation_data, region=None, level=0, categories: list=[1, 2]):

    slide = openslide.open_slide(os.path.join(MIDOG_DEFAULT_PATH, image_filename))
    level = 0
    down_factor = slide.level_downsamples[level]

    if region is None:
        x0, y0 = 0, 0
        width, height = slide.level_dimensions[level]
    else:
        x0, y0 = region[:2]
        width, height = region[2:] - region[:2]

    img_region = np.array(slide.read_region(location=(int(x0 * down_factor),int(y0 * down_factor)),
                                          level=level, size=(width, height)))[:, :, :3]
    
    bboxes, labels = get_bboxes(image_filename, annotation_data, categories)
    bboxes, labels = filter_bboxes(bboxes, labels, x0, y0, width, height)
                                          
    return img_region, {'boxes': bboxes, 'labels': labels}

    