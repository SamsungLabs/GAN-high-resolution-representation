import numpy as np
import torch
from lib.nn.networks_stylegan2 import Generator
from lib.nn.layers.modifiers import apply_lr_mult, apply_wscale
from lib.nn.layers.weight_init import Normal


class ImageGenerator():

    def __init__(self, gpu_ids, gan_dir, gan='ffhq', batch_size=16, return_latents=False, use_latent_avg=True):
        super().__init__()

        max_res_log2_dict = {'ffhq': 10, 'church': 9, 'cat': 9, 'car': 9, 'horse': 9, 'cityscapes': 9}
        self.max_res_log2 = max_res_log2_dict[gan]

        self.return_latents = return_latents
        self.batch_size = batch_size

        ngpus = len(gpu_ids)
        self.device = torch.device(gpu_ids[0] if (torch.cuda.is_available() and ngpus > 0) else 'cpu')
        self.cfg = self._get_config(max_res_log2=self.max_res_log2)
        self.latent_size = self.cfg['latent_size']

        self.netG = self.init_generator(self.cfg, initialize=False)
        stylegan_name = f'{gan}.tar'
        checkpoint = torch.load(f'{gan_dir}/{stylegan_name}')
        self.netG.load_state_dict(checkpoint['netGA'])
        self.netG.eval()
        self.netG = self.netG.to(self.device)
        self.latent_avg = checkpoint['latent_avg'].to(self.device)

    def init_generator(self, cfg, initialize=False):

        netG = Generator(cfg['max_res_log2'], latent_size=cfg['latent_size'], fmap_base=cfg['fmap_base'],
                         fmap_max=cfg['fmap_max'],
                         base_scale_h=cfg['base_scale_y'], base_scale_w=cfg['base_scale_x'],
                         channels=cfg['channels'], use_activation=False, use_pn=True)

        mapping_lr_mult = 0.01
        if mapping_lr_mult != 1.0:
            apply_lr_mult(netG.mapping, lr_mult=mapping_lr_mult, weight_name='weight')
            apply_lr_mult(netG.mapping, lr_mult=mapping_lr_mult, weight_name='bias')
        apply_wscale(netG, gain=1.)

        if initialize:
            if mapping_lr_mult != 1.0:
                scale = 1 / mapping_lr_mult
                netG.mapping.apply(Normal(scale))
            netG.apply(Normal(1.0))

        return netG

    def _get_config(self, max_res_log2=9):
        cfg = {}

        # network parameters
        cfg['use_wscale'] = True

        cfg['fmap_base'] = 2*8192
        cfg['fmap_decay'] = 1.0
        cfg['fmap_max'] = 512
        cfg['max_res_log2'] = max_res_log2
        cfg['fix_noise'] = False

        cfg['base_scale_x'] = 4
        cfg['base_scale_y'] = 4

        # input format
        cfg['latent_size'] = 512

        cfg['channels'] = 3
        cfg['imrange'] = (-1, 1)

        return cfg

    def _transform_gan_back(self, img, cfg):
        imrange = cfg['imrange']
        channel_swap = (0, 2, 3, 1)
        img = np.transpose(img, axes=channel_swap)
        img = (img - imrange[0]) / (imrange[1] - imrange[0])
        img = np.clip(img, 0.0, 1.0)
        img = 255. * img
        img = img.astype(np.uint8)
        return img

    def get_images(self, n):

        n_batches = n // self.batch_size
        n_batches += 1 if n % self.batch_size > 0 else 0

        n_generated = 0
        for n_batch in range(n_batches):
            batch_size_s = min(self.batch_size, n - n_generated)
            latent_z = torch.randn(size=(batch_size_s, self.latent_size)).to(self.device)
            with torch.no_grad():
                data, features = self.netG(latent_z, latent_avg=self.latent_avg)
            data = data.detach().cpu().numpy()
            latent_z_np = latent_z.cpu().numpy()
            features = [f.detach().cpu().numpy() for f in features]

            imgs = self._transform_gan_back(data, self.cfg)
            n_generated += imgs.shape[0]
            for i in range(imgs.shape[0]):
                img = imgs[i]
                feats = [feat[i] for feat in features]
                latent = latent_z_np[i]
                if self.return_latents:
                    yield img, feats, latent
                else:
                    yield img, feats