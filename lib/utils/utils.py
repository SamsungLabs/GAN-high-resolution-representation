from os.path import join, isdir, basename, splitext, isfile, islink
from os.path import sep as os_path_sep
from os import listdir, walk
import numpy as np
import cv2
import yaml


def list_subdirs(base_dir):
    subdirs = []
    for f in listdir(base_dir):
        if not isdir(join(base_dir, f)):
            continue
        subdirs.append(f)
    return subdirs


def list_files_with_ext(base_dir, valid_exts, recursive=False):
    images = []

    if recursive:
        list_files_with_ext_rec(base_dir, images, valid_exts)
    else:
        assert isdir(base_dir) or islink(base_dir), f'{base_dir} is not a valid directory'
        base_path_len = len(base_dir.split(os_path_sep))
        for root, dnames, fnames in sorted(walk(base_dir)):
            root_parts = root.split(os_path_sep)
            root_m = os_path_sep.join(root_parts[base_path_len:])
            for fname in fnames:
                if not isfile(join(root, fname)):
                    continue
                filext = splitext(fname.lower())[1]
                if filext not in valid_exts:
                    continue
                path = join(root_m, fname)
                images.append(path)

    return images


def list_files_with_ext_rec(base_dir, images, valid_exts):
    assert isdir(base_dir), f'{base_dir} is not a valid directory'
    base_path_len = len(base_dir.split(os_path_sep))
    for root, dnames, fnames in sorted(walk(base_dir, followlinks=True)):
        root_parts = root.split(os_path_sep)
        root_m = os_path_sep.join(root_parts[base_path_len:])

        for fname in fnames:
            if not isfile(join(root, fname)):
                continue
            filext = splitext(fname.lower())[1]
            if filext not in valid_exts:
                continue
            path = join(root_m, fname)
            images.append(path)


def list_images(base_dir, valid_exts=['.jpg', '.jpeg', '.png', '.bmp', '.ppm']):
    images_list = []
    for f in listdir(base_dir):
        if not isfile(join(base_dir, f)):
            continue
        filext = splitext(f.lower())[1]
        if filext not in valid_exts:
            continue
        images_list.append(f)
    return images_list


def get_seg_color_map():
    color_bg = np.array([0,0,0],dtype=np.uint8)
    color_fg = np.array([13, 198, 20],dtype=np.uint8)
    color_neg = np.array([54, 30, 211],dtype=np.uint8)
    color_map = []
    color_map.append([0, color_bg])
    color_map.append([1, color_fg])
    color_map.append([2, color_neg])
    return color_map


def get_draw_mask(img, mask, alpha=0.5, color_map=None, skip_background=True):
    if color_map is None:
        color_map = get_seg_color_map()

    im_cpy = np.array(img)

    im_cpy_b = im_cpy[:, :, 0]
    im_cpy_g = im_cpy[:, :, 1]
    im_cpy_r = im_cpy[:, :, 2]

    for idx, color in color_map:
        if idx == 0 and skip_background:
            continue
        mask_cur = mask == idx
        im_cpy_b[mask_cur] = alpha * color[0] + (1 - alpha) * im_cpy_b[mask_cur]
        im_cpy_g[mask_cur] = alpha * color[1] + (1 - alpha) * im_cpy_g[mask_cur]
        im_cpy_r[mask_cur] = alpha * color[2] + (1 - alpha) * im_cpy_r[mask_cur]

    im_cpy[:, :, 0] = im_cpy_b
    im_cpy[:, :, 1] = im_cpy_g
    im_cpy[:, :, 2] = im_cpy_r

    return im_cpy


def morph_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def load_config_file(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)
    return cfg


def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)
    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.
    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return xywh[0], xywh[1], xywh[0] + w, xywh[1] + h
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))


def bbox_xyxy_to_xywh(xyxy):
    """Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h).
    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (x, y, w, h).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.
    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1 + 1, xyxy[3] - y1 + 1
        return x1, y1, w, h
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))


def calculate_xyxy_area(xyxy):
    x0, y0, x1, y1 = xyxy
    return abs(x0 - x1) * abs(y0 - y1)


def draw_rects(img, xyxys, color=(11, 250, 12), thickness=2):
    for xyxy in xyxys:
        draw_rect(img, xyxy, color=color, thickness=thickness)


def draw_rect(img, xyxy, color=(11, 250, 12), thickness=2, **kwargs):

    xyxy[0] = min(img.shape[1], max(0, xyxy[0]))
    xyxy[1] = min(img.shape[0], max(0, xyxy[1]))
    xyxy[2] = min(img.shape[1], max(0, xyxy[2]))
    xyxy[3] = min(img.shape[0], max(0, xyxy[3]))

    pt1 = (xyxy[0], xyxy[1])
    pt2 = (xyxy[2], xyxy[3])

    cv2.rectangle(img, pt1, pt2, color, thickness=thickness, **kwargs)


def crop_image(img, bbox):
    x_st = bbox[0]
    y_st = bbox[1]

    x_en = bbox[0] + bbox[2] - 1
    y_en = bbox[1] + bbox[3] - 1

    x_st_pad = int(max(0, -x_st))
    y_st_pad = int(max(0, -y_st))
    x_en_pad = int(max(0, x_en - img.shape[1] + 1))
    y_en_pad = int(max(0, y_en - img.shape[0] + 1))

    x_en = x_en + max(0, -x_st)
    y_en = y_en + max(0, -y_st)
    x_st = max(0, x_st)
    y_st = max(0, y_st)

    if y_st_pad != 0 or y_en_pad != 0 or x_st_pad != 0 or x_en_pad != 0:
        assert len(img.shape) in (2, 3)
        if len(img.shape) == 3:
            img_pad = np.zeros((img.shape[0]+y_st_pad+y_en_pad, img.shape[1]+x_st_pad+x_en_pad, img.shape[2]), dtype=img.dtype)
            img_pad[y_st_pad:y_st_pad+img.shape[0], x_st_pad:x_st_pad+img.shape[1], :] = img
        else:
            img_pad = np.zeros((img.shape[0]+y_st_pad+y_en_pad, img.shape[1]+x_st_pad+x_en_pad), dtype=img.dtype)
            img_pad[y_st_pad:y_st_pad+img.shape[0], x_st_pad:x_st_pad+img.shape[1]] = img
    else:
        img_pad = img
    img_cropped = img_pad[y_st:y_en+1, x_st:x_en+1]
    return img_cropped


def prepare_crop(im, prepare_sz, fit_whole=False, use_nn_interpolation=False):
    if im.shape[0] != prepare_sz[1] or im.shape[1] != prepare_sz[0]:
        prepare_r = float(prepare_sz[0]) / prepare_sz[1]
        orig_r = float(im.shape[1]) / im.shape[0]

        if fit_whole:
            do_fit_width = orig_r > prepare_r
        else:
            do_fit_width = orig_r < prepare_r

        if do_fit_width:
            # fit width
            crop_w = im.shape[1]
            crop_h = crop_w / prepare_r
        else:
            # fit height
            crop_h = im.shape[0]
            crop_w = crop_h * prepare_r

        crop_x = int((im.shape[1] - crop_w) / 2.)
        crop_y = int((im.shape[0] - crop_h) / 2.)
        crop_w = int(crop_w)
        crop_h = int(crop_h)

        crop_rect = [crop_x, crop_y, crop_w, crop_h]
        im = crop_image(im, crop_rect)

        interp = cv2.INTER_NEAREST if use_nn_interpolation else cv2.INTER_LINEAR
        im = cv2.resize(im, prepare_sz, interpolation=interp)
    return im