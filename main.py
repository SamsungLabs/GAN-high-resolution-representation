import time
import numpy as np
from lib.core.seg_annotator import SegmentationAnnotator
from lib.utils.utils import load_config_file
import tkinter as tk


if __name__ == '__main__':

    seed = int(time.time())
    cfg = load_config_file('configs/config.yml')
    root_dir = cfg['BASE_DIR']
    gan = cfg['GAN']
    gan_dir = cfg['GAN_DIR']
    gan_gpu_ids = cfg['GAN_GPU_IDS']
    solver_gpu_ids = cfg['SOLVER_GPU_IDS']
    annotation = cfg['ANNOTATION']
    no_gan = cfg.get('NO_GAN', False)
    imgs_dir = cfg.get('IMGS_DIR', None)

    np.random.seed(seed)

    root = tk.Tk()
    if annotation == 'segmentation':
        SegmentationAnnotator(root, root_dir, gan_gpu_ids=gan_gpu_ids, solver_gpu_ids=solver_gpu_ids,
                              gan_dir=gan_dir, gan=gan).pack(fill='both', expand=True)
    else:
        print(f'uknown annotation type: {annotation}')
    root.mainloop()