from torch.utils.data import Dataset
from os.path import join, splitext
import cv2
import numpy as np
import time
import pickle
import random
from lib.utils.utils import list_files_with_ext


class CollectionDataset(Dataset):

    def __init__(self, db_dir, cfg, is_validation=False, output_idx=False, max_samples=None,
                 allow_missed_mask=False, load_to_memory=True, min_epoch_size=5):
        self._cfg = cfg
        self._is_train = not is_validation
        self._output_idx = output_idx
        self._max_samples = max_samples
        self._allow_missed_mask = allow_missed_mask
        self._preprocess_mask = cfg['preprocess_mask']
        self._not_ignore_classes = cfg['not_ignore_classes']
        self._load_to_memory = load_to_memory
        self._db_dir = db_dir
        self._min_epoch_size = min_epoch_size

        print('loading data..')
        tic = time.time()
        self._load_data()
        print('loading finished in {:.3} sec'.format(time.time() - tic))

    def _load_data(self):
        db_dir = self._db_dir

        samples = []

        feat_names = list_files_with_ext(db_dir, valid_exts=['.pickle'])
        feat_names = [f for f in feat_names if 'feat' in f]

        if self._max_samples is not None:
            self._max_samples = min(len(feat_names), self._max_samples)
            feat_names = random.sample(feat_names, self._max_samples)

        if self._load_to_memory:
            for feat_name in feat_names:
                samples.append(self.load_sample(feat_name))
            self._samples = samples
        else:
            self._samples = None
        self._feat_names = feat_names
        self._db_dir = db_dir

    def load_sample(self, feature_name):

        db_dir = self._db_dir

        imbase = splitext(feature_name)[0]
        imname = imbase.replace('feat', 'img') + '.jpg'
        mask_name = imbase.replace('feat', 'mask') + '.png'

        img_data = cv2.imread(join(db_dir, imname))
        img_data = img_data[:, :, [2, 1, 0]]  # bgr to rgb

        mask_data = cv2.imread(join(db_dir, mask_name), 0)
        if mask_data is None and self._allow_missed_mask:
            mask_data = np.zeros((img_data.shape[0], img_data.shape[1]))
        assert (mask_data is not None)

        with open(join(db_dir, feature_name), 'rb') as fp:
            features = pickle.load(fp)

        return (mask_data, img_data, features)

    def get_item(self, idx):

        if self._load_to_memory:
            mask, img, features = self._samples[idx]
        else:
            mask, img, features = self.load_sample(self._feat_names[idx])

        if self._preprocess_mask:
            if not self._is_train:

                mask_cpy = np.array(mask, copy=True)
                mask_false = np.logical_and(mask_cpy >= 64, mask_cpy <= 192)
                mask_ignore = mask_cpy < 64
                mask_true = mask_cpy > 192

                mask = mask.astype(np.int32)

                mask[mask_true] = 1
                mask[mask_false] = 0
                mask[mask_ignore] = -1

            else:
                mask_cpy = np.array(mask, copy=True)
                mask_false = np.logical_and(mask_cpy >= 64, mask_cpy <= 192)
                mask_ignore = mask_cpy < 64
                mask_true = mask_cpy > 192

                mask = mask.astype(np.int32)

                mask[mask_true] = 1
                mask[mask_false] = 0
                mask[mask_ignore] = -1
        else:
            mask = mask.astype(np.int32)

        if self._not_ignore_classes is not None:
            test_elements = self._not_ignore_classes
            mask_not_ignore = np.isin(mask, test_elements)
            mask[np.logical_not(mask_not_ignore)] = -1

        mask = mask[:, :, np.newaxis]
        img = img.astype(np.float32)

        channel_swap = (2, 0, 1)
        mask = np.transpose(mask, axes=channel_swap)
        img = np.transpose(img, axes=channel_swap)

        data = {}
        data['img'] = img
        data['mask'] = mask
        data['features'] = features

        return data

    def __getitem__(self, idx):

        if len(self._feat_names) < self._min_epoch_size:
            idx = np.random.randint(len(self._feat_names))

        return self.get_item(idx)

    def get_imname(self, idx):
        feat_name = self._feat_names[idx]
        feat_namebase = splitext(feat_name)[0]
        imname = feat_namebase.replace('feat', 'img') + '.jpg'
        return imname

    def __len__(self):
        return max(self._min_epoch_size, len(self._feat_names))

