from utils.summary import Summary
from utils.logger import Logger
import numpy as np
import os
import torch
import torch.utils.data
from models import *
from datasets.coco import COCO, COCO_eval
from warmup_scheduler import GradualWarmupScheduler
from utils.tools import _tranpose_and_gather_feature
from losses import RegLoss, NegLoss
from utils.post_proc import ctdet_decode
from utils.image import transform_preds
from tqdm import tqdm
import shutil


class Trainer:
    def __init__(self, config):
        self._cuda = config.cuda & torch.cuda.is_available()
        self._logger = Logger(config.logdir)
        self._summary = Summary(config.sumdir)
        self._log_interval = config.log_interval
        self._epochs = config.epochs
        self._ckpt_dir = config.ckpt_dir
        self._test_topk = config.test_topk
        self.num_classes = config.num_classes
        self.metrics = {'mAP': 0, 'train_loss': float('inf'), 'best_model_epoch': 0}

        if not os.path.exists(self._ckpt_dir):
            os.makedirs(self._ckpt_dir)

        self._reg_loss = RegLoss()
        self._neg_loss = NegLoss()

        torch.manual_seed(317)
        if config.cuda:
            torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training
        train_dataset = COCO(config.data_dir, 'train', split_ratio=config.split_ratio, img_size=config.img_size)
        self._train_loader = torch.utils.data.DataLoader(train_dataset,
                                                         batch_size=config.batch_size,
                                                         shuffle=True,
                                                         num_workers=config.num_workers,
                                                         pin_memory=True,
                                                         drop_last=True,
                                                         sampler=None)

        self._val_dataset = COCO_eval(config.data_dir, 'val', test_scales=[1.], test_flip=False)
        self._val_loader = torch.utils.data.DataLoader(self._val_dataset, batch_size=1,
                                                       shuffle=False, num_workers=1, pin_memory=True,
                                                       collate_fn=self._val_dataset.collate_fn)

        if 'hourglass' in config.arch:
            self._model = get_hourglass(config.arch, num_classes=config.num_classes)
        elif 'resdcn' in config.arch:
            self._model = get_resdcn(num_layers=int(config.arch[6:]), num_classes=config.num_classes)
        else:
            raise NotImplementedError

        self._optimizer = torch.optim.Adam(self._model.parameters(), config.lr)
        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
                                                                        T_max=50,
                                                                        eta_min=1e-6)  # 1e-6
        self.scheduler_warmup = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=3,
            after_scheduler=self._lr_scheduler)

        if self._cuda:
            self._model.cuda()

    def train(self, epoch):
        self._model.train()
        torch.cuda.empty_cache()
        train_losses = .0
        hmap_losses = .0
        reg_losses = .0
        w_h_losses = .0
        with self._logger as logger:
            for batch_idx, batch in enumerate(tqdm(self._train_loader,
                                                   desc="Training [{}/{}]".format(
                                                       epoch,
                                                       self._epochs))):
                for k in batch:
                    if k != 'meta' and self._cuda:
                        batch[k] = batch[k].cuda(non_blocking=True)

                outputs = self._model(batch['image'])
                hmap, regs, w_h_ = zip(*outputs)
                regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
                w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]

                hmap_loss = self._neg_loss(hmap, batch['hmap'])
                reg_loss = self._reg_loss(regs, batch['regs'], batch['ind_masks'])
                w_h_loss = self._reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])
                loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                train_losses += loss.item()
                hmap_losses += hmap_loss.item()
                reg_losses += reg_loss.item()
                w_h_losses += w_h_loss.item()

                if batch_idx % self._log_interval == 0:
                    logger.debug('[%d/%d-%d/%d] ' % (epoch, self._epochs, batch_idx, len(self._train_loader)) +
                                 ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f' %
                                 (hmap_loss.item(), reg_loss.item(), w_h_loss.item()))

                    step = len(self._train_loader) * epoch + batch_idx
                    self._summary.add_scalar('hmap_loss', hmap_loss.item(), step)
                    self._summary.add_scalar('reg_loss', reg_loss.item(), step)
                    self._summary.add_scalar('w_h_loss', w_h_loss.item(), step)

        return {'train_loss': train_losses / len(self._train_loader),
                'hmap_loss': hmap_losses / len(self._train_loader),
                'reg_loss': reg_losses / len(self._train_loader),
                'w_h_loss': w_h_losses / len(self._train_loader),
                'epoch': epoch}

    @torch.no_grad()
    def validation(self, epoch):
        self._model.eval()
        torch.cuda.empty_cache()
        max_per_image = 100

        results = {}

        for inputs in tqdm(self._val_loader, desc="Validating [{}/{}]".format(
                epoch,
                self._epochs)):
            img_id, inputs = inputs[0]
            detections = []
            for scale in inputs:
                if self._cuda:
                    inputs[scale]['image'] = inputs[scale]['image'].cuda()
                output = self._model(inputs[scale]['image'])[-1]

                dets = ctdet_decode(*output, K=self._test_topk)
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                top_preds = {}
                dets[:, :2] = transform_preds(dets[:, 0:2],
                                              inputs[scale]['center'],
                                              inputs[scale]['scale'],
                                              (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                               inputs[scale]['center'],
                                               inputs[scale]['scale'],
                                               (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                clses = dets[:, -1]
                for j in range(self.num_classes):
                    inds = (clses == j)
                    top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                    top_preds[j + 1][:, :4] /= scale

                detections.append(top_preds)

            bbox_and_scores = {j: np.concatenate([d[j] for d in detections], axis=0)
                               for j in range(1, self.num_classes + 1)}
            scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, self.num_classes + 1)])
            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, self.num_classes + 1):
                    keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                    bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

            results[img_id] = bbox_and_scores

        eval_results = self._val_dataset.run_eval(results, save_dir=self._ckpt_dir)
        self._logger.log(eval_results, "debug")
        self._summary.add_scalar('val_mAP/mAP', eval_results[0], epoch)
        return eval_results[0]

    def _on_epoch_finish(self, epoch_result):
        self._logger.log('[{}/{}], train_loss: {:.4f}, hmap_loss: {:.4f}, reg_loss: {:.4f}, w_h_loss: {:.4f},'.format(
            epoch_result['epoch'], self._epochs,
            epoch_result['train_loss'], epoch_result['hmap_loss'],
            epoch_result['reg_loss'], epoch_result['w_h_loss']))

        net_save_path = os.path.join(self._ckpt_dir, "model_latest.pth")
        net_save_path_best = os.path.join(self._ckpt_dir, "model_best.pth")

        self._save_checkpoint(epoch_result['epoch'], net_save_path)
        save_best = False
        if self._val_loader is not None:  # 使用mAP作为最优模型指标
            mAP = self.validation(epoch_result['epoch'])
            if mAP >= self.metrics['mAP']:
                save_best = True
                self.metrics['train_loss'] = epoch_result['train_loss']
                self.metrics['mAP'] = mAP
                self.metrics['best_model_epoch'] = epoch_result['epoch']
        else:
            if epoch_result['train_loss'] <= self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = epoch_result['train_loss']
                self.metrics['best_model_epoch'] = epoch_result['epoch']
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.6f}, '.format(k, v)
        self._logger.log(best_str)

        if save_best:
            shutil.copy(net_save_path, net_save_path_best)
            self._logger.log("Saving current best: {}".format(net_save_path_best))
        else:
            self._logger.log("Saving checkpoint: {}".format(net_save_path))

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self._logger.log('{}:{}'.format(k, v))
        self._logger.log('finish train')

    def _save_checkpoint(self, epoch, file_name):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state_dict = self._model.state_dict()
        state = {
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self._optimizer.state_dict(),
            'scheduler': self._lr_scheduler.state_dict(),
            'metrics': self.metrics
        }
        torch.save(state, file_name)

    def runner(self):
        for epoch in range(1, self._epochs + 1):
            epoch_result = self.train(epoch)
            self._on_epoch_finish(epoch_result=epoch_result)
            self.scheduler_warmup.step()
        self._on_train_finish()

