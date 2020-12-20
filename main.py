import anyconfig
import os

# nohup python main.py  > log.out 2>&1 &
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# CUDA: 0

class Config:
    logdir = "logs/coco_hg_512_d/"
    sumdir = "summary/coco_hg_512_d/"
    data_dir = "./coco_person/"
    split_ratio = 1.0
    img_size = 512
    batch_size = 25
    num_workers = 4
    arch = "small_hourglass"
    num_classes = 1
    lr = 5e-4
    cuda = True
    log_interval = 100
    epochs = 100
    ckpt_dir = "ckpt/coco_hg_512_dp/"
    test_topk = 100


def main(config):
    from runner.trainer import Trainer
    trainer = Trainer(config)
    trainer.runner()


if __name__ == '__main__':
    main(Config)
