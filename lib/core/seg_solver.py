import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from os.path import join, splitext
from lib.utils.utils import list_files_with_ext
import logging
logging.getLogger().setLevel(logging.INFO)

from lib.nn.networks_seg_2 import Decoder
from lib.data.dataset import CollectionDataset
from lib.metrics.metrics import SegmentationMetric
from lib.nn.layers.weight_init import Normal
from lib.metrics.metrics import Accuracy, SegmentationMetric
from lib.nn.layers.modifiers import apply_wscale


class SegSolver():
    def __init__(self, max_res_log2, path_to_data, checkpoints_dir, gpu_ids):

        self.path_to_data = path_to_data
        self.checkpoints_dir = checkpoints_dir

        ngpus = len(gpu_ids)
        device = torch.device(gpu_ids[0] if (torch.cuda.is_available() and ngpus > 0) else 'cpu')

        self.is_trained = False
        self.params_file = None
        self.device = device
        self.max_res_log2 = max_res_log2
        self.cfg = self.get_config(max_res_log2=max_res_log2)
        self.net = self.init_net(self.max_res_log2)
        self.is_trained = self.load()

    def init_net(self, max_res_log2):

        # net = Decoder(self.cfg)
        # net = Decoder(self.cfg)
        # net.apply(XavierGluon(rnd_type='gaussian', factor_type='in', magnitude=2.34))

        net = Decoder(max_res_log2=9, fmap_base=2 * 128 * 64, num_classes=2,
                      decoder_fmap_max=128, use_bn=False, dropout=0.1, skip_n_last_features=2)
        apply_wscale(net, gain=1.)
        net.apply(Normal(1.0))

        self.print_params(net, 'decoder')

        net.train()
        net = net.to(self.device)

        return net

    def init_trainer(self, net):

        optimizer, optimizer_params = self.get_optimizer_params()
        loss = nn.CrossEntropyLoss(ignore_index=-1)

        if optimizer == 'adam':
            trainer = torch.optim.Adam(net.parameters(), **optimizer_params)
        elif optimizer == 'sgd':
            trainer = torch.optim.SGD(net.parameters(), **optimizer_params)
        else:
            logging.info(f'unknown optimizer: {optimizer}')
            raise ValueError

        return trainer, loss

    def print_params(self, net, name='', verbose=False):
        logging.info('-----------------------------------------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            logging.info(net)
        logging.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        logging.info('-----------------------------------------------')

    def get_config(self, max_res_log2=9):
        cfg = {}

        cfg['seed'] = 1
        cfg['kvstore'] = 'nccl'
        cfg['cache_max_size'] = 4  # in GB
        cfg['plot_graph'] = True

        cfg['num_classes'] = 2
        cfg['not_ignore_classes'] = None
        cfg['cls_type'] = 'hair'

        cfg['train_epochs'] = 24

        # cfg['base_lr'] = 0.01
        # cfg['factor_d'] = 0.1
        # cfg['wd'] = 1e-3
        # cfg['optimizer'] = 'sgd'
        # cfg['momentum'] = 0.9
        # cfg['epochs_steps'] = [cfg['train_epochs'] * 0.6, cfg['train_epochs'] * 0.8]
        # cfg['scheduler'] = 'steps'

        cfg['base_lr'] = 1e-3
        print(cfg['base_lr'])
        cfg['factor_d'] = 0.1
        cfg['wd'] = None
        cfg['optimizer'] = 'adam'
        cfg['momentum'] = None

        cfg['preprocess_mask'] = True

        cfg['train_display_iters'] = 4
        cfg['train_batch_size'] = 1
        cfg['val_batch_size'] = 1

        cfg['val_loader_workers'] = 0
        cfg['train_loader_workers'] = 0

        cfg['train_show_images'] = 1
        cfg['val_show_images'] = 1

        cfg['val_report_intermediate'] = False
        cfg['val_report_interval'] = 0.34

        cfg['use_bn'] = True
        cfg['use_sync_bn'] = False
        cfg['use_dropout'] = True
        cfg['start_res'] = 0

        cfg['features'] = [32, 32, 32, 32, 32, 32, 32, 32, 16]
        cfg['in_channels'] = [512, 512, 512, 512, 512, 256, 128, 64, 32]

        cfg['features'] = cfg['features'][:max_res_log2-1] + [cfg['num_classes']]
        cfg['in_channels'] = cfg['in_channels'][:max_res_log2-1]

        cfg['dtype'] = 'fp32'

        return cfg

    def init_data(self):

        cfg = self.cfg
        train_dataset = CollectionDataset(self.path_to_data, cfg, max_samples=None, load_to_memory=False)
        train_n_samples = len(train_dataset)
        if train_n_samples <= 0:
            print('number of training samples should be > 0')
            raise ValueError

        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['train_batch_size'], drop_last=True,
                                                 shuffle=True, num_workers=cfg['train_loader_workers'])

        iters_per_epoch = int(len(train_dataset) / cfg['train_batch_size'])


        print('total train samples: {}'.format(train_n_samples))
        print('batch size: {}'.format(cfg['train_batch_size']))
        print('epoch size: {}'.format(iters_per_epoch))

        return train_dataset, train_data, iters_per_epoch

    def init_eval_data(self, input_dir):

        cfg = self.cfg
        eval_dataset = CollectionDataset(input_dir, cfg, max_samples=None, load_to_memory=False, output_idx=True)
        eval_n_samples = len(eval_dataset)
        if eval_n_samples <= 0:
            print('number of training samples should be > 0')
            raise ValueError

        eval_data = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg['val_batch_size'], drop_last=True,
                                                shuffle=True, num_workers=cfg['val_loader_workers'])

        print('total eval samples: {}'.format(eval_n_samples))
        print('batch size: {}'.format(cfg['val_batch_size']))

        return eval_dataset, eval_data

    def init_metric(self):
        train_metric = Accuracy()
        return train_metric

    def get_optimizer_params(self):

        cfg = self.cfg

        base_lr = cfg['base_lr']
        optimizer = cfg['optimizer']
        wd = cfg['wd']
        momentum = cfg['momentum']

        optimizer_params = []
        if base_lr is not None:
            optimizer_params.append(('lr', base_lr))
        if momentum is not None:
            optimizer_params.append(('momentum', momentum))
        if wd is not None:
            optimizer_params.append(('wd', wd))

        optimizer_params = {name: value for name, value in optimizer_params}

        return optimizer, optimizer_params

    def evaluate(self, input_dir, output_dir=None):
        eval_dataset, eval_dataloader = self.init_eval_data(input_dir)
        eval_metric = SegmentationMetric(self.cfg['num_classes'])
        loss_f = nn.CrossEntropyLoss(ignore_index=-1)
        return self.evaluate_for_data(eval_dataloader, loss_f, eval_metric)

    def evaluate_for_data(self, val_dataloader, loss_f, eval_metric):
        total_loss = 0
        total_cnt = 0
        for batch in val_dataloader:
            mask = batch['mask'].to(torch.long).to(self.device)
            features = batch['features']
            features = [f.to(self.device) for f in features]

            outputs = self.net(features)

            ce_loss = loss_f(outputs, mask)

            loss_v = ce_loss.item()
            total_loss += loss_v
            total_cnt += 1

            h = 2 ** self.max_res_log2
            w = 2 ** self.max_res_log2

            if outputs.shape[2] != h or outputs.shape[3] != w:
                outputs = torch.nn.functional.interpolate(outputs, (h, w), mode='bilinear',
                                                          align_corners=True)

            eval_metric.update(mask, outputs)

        if total_cnt > 0:
            total_loss = total_loss / total_cnt
        else:
            total_loss = 0.0

        pixAcc, mIoU = eval_metric.get()

        result = []
        result.append(('pixAcc', pixAcc))
        result.append(('mIoU', mIoU))
        result.append(('total-loss', total_loss))

        return result

    def predict(self, features):

        features_n = []
        for f in features:
            if not isinstance(f, torch.Tensor):
                f = torch.from_numpy(f).to(torch.float32)
            if len(f.shape) == 3:
                f = f.unsqueeze(0)
            features_n.append(f)

        features = [f.to(self.device) for f in features_n]
        with torch.no_grad():
            pred_mask = self.net(features)

        h = 2 ** self.max_res_log2
        w = 2 ** self.max_res_log2

        if pred_mask.shape[2] != h or pred_mask.shape[3] != w:
            pred_mask = torch.nn.functional.interpolate(pred_mask, (h, w), mode='bilinear',
                                                        align_corners=True)
        probs = torch.softmax(pred_mask, dim=1).detach().cpu().numpy()

        pred_masks = torch.argmax(pred_mask, dim=1, keepdim=True).detach().cpu().numpy()
        pred_masks = np.transpose(pred_masks, (0, 2, 3, 1))

        return pred_masks, probs

    def save(self, suffix=None):
        if suffix is None:
            param_name = 'checkpoint_latest.tar'
        else:
            param_name = f'checkpoint_{suffix}.tar'
        self.params_file = param_name

        checkpoint_states = {'net': self.net.state_dict()}
        torch.save(checkpoint_states, join(self.checkpoints_dir, param_name))

    def load(self):

        params_files = list_files_with_ext(self.checkpoints_dir, valid_exts=['.tar'])
        if len(params_files) > 0:
            params_n = [p for p in params_files if 'train_number' in p]
            if len(params_n) > 0:
                params_n = [(p, int(splitext(p)[0].split('_')[-1])) for p in params_n]
                params_n = sorted(params_n, key=lambda x:x[1], reverse=True)
                params_file = params_n[0][0]
            else:
                params_file = params_files[0]
            print(f'loading checkpoint: {params_file}')
            self.params_file = params_file
            checkpoint = torch.load(join(self.checkpoints_dir, params_file))
            self.net.load_state_dict(checkpoint['net'])
            return True
        else:
            return False

    def fit(self, epoch_end_callback=None):

        self.net = self.init_net(self.max_res_log2)
        self.train_dataset, self.train_dataloader, self.iters_per_epoch = self.init_data()
        self.trainer, self.loss = self.init_trainer(self.net)
        self.train_metric = self.init_metric()

        trainer = self.trainer
        train_dataloader = self.train_dataloader
        iters_per_epoch = self.iters_per_epoch
        batch_size = self.cfg['train_batch_size']

        display = self.cfg['train_display_iters']
        train_metric = self.train_metric
        epochs_to_train = self.cfg['train_epochs']

        scores = []

        for epoch in range(epochs_to_train):
            # new epoch
            tic = time.time()

            train_metric.reset()

            nbatch = 0
            speed_tic = time.time()

            for batch in train_dataloader:

                mask = batch['mask'].to(torch.long).to(self.device)
                features = batch['features']
                features = [f.to(self.device) for f in features]

                outputs = self.net(features)

                h = 2 ** self.max_res_log2
                w = 2 ** self.max_res_log2
                if outputs.shape[2] != h or outputs.shape[3] != w:
                    outputs = torch.nn.functional.interpolate(outputs, (h, w), mode='bilinear',
                                                              align_corners=True)

                mask = mask.view((mask.size(0), -1))
                outputs = outputs.view((outputs.size(0), outputs.size(1), -1))
                ce_loss = self.loss(outputs, mask)

                train_metric.update(mask, torch.argmax(outputs, dim=1))

                trainer.zero_grad()
                ce_loss.backward()
                trainer.step()

                nbatch += 1
                global_step = (epoch * iters_per_epoch + nbatch) * batch_size

                # speedometer
                if display is not None and nbatch % display == 0:

                    speed = 1.0 * display * batch_size / (time.time() - speed_tic)

                    accuracy = train_metric.get()
                    total_loss = ce_loss.item()
                    train_metric.reset()

                    msg = f'Epoch[{epoch}] Batch[{nbatch}] Speed: {speed:.2f} samples/sec Loss: {total_loss:4f} Accuracy: {accuracy:.1f}'
                    logging.info(msg)
                    speed_tic = time.time()

            # global_step = ((epoch + 1) * iters_per_epoch) * batch_size

            # # one epoch of training is finished
            # for name, val in name_values:
            #     logging.info('Epoch[%d] Train-%s=%f', epoch + 1, name, val)
            time_cost = (time.time() - tic)
            logging.info('Epoch[%d] Time cost=%.3f', epoch + 1, time_cost)

            if epoch_end_callback is not None:
                epoch_end_callback()

        # save checkpoint
        self.is_trained = True
        self.save()
        # self.save(suffix=f'train_number_{len(self.train_dataset)}')

        return scores