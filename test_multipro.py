# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset, ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
import cv2

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu_id, result_queue):
    segmentation_module.eval()

    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        img_resized_list = batch_data['img_data']
        segSize = (batch_data['img_ori'].shape[0], batch_data['img_ori'].shape[1])

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
        
        relpath = os.path.relpath(batch_data['info'], cfg.data_dir).replace('.jpg','.png')
        tgt_path = os.path.join(cfg.out_dir,  relpath)
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
        cv2.imwrite(tgt_path, pred.astype(np.uint16))

    result_queue.put_nowait(0)


def worker(cfg, gpu_id, start_idx, end_idx, result_queue):
    torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test[start_idx: end_idx],
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_test, cfg, gpu_id, result_queue)


def main(cfg, gpus):
    lines = cfg.list_test
    num_files = len(lines)

    num_files_per_gpu = math.ceil(num_files / len(gpus))

    pbar = tqdm(total=num_files)

    result_queue = Queue(500)
    procs = []
    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)
        proc = Process(target=worker, args=(cfg, gpu_id, start_idx, end_idx, result_queue))
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        _ = result_queue.get()
        processed_counter += 1
        pbar.update(1)

    for p in procs:
        p.join()

    print('Export Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Export"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="path to root of dataset",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="path to root of output",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_list(["data_dir", args.data_dir, "out_dir", args.out_dir])
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # generate testing image list
    imgs = []
    data_dir = args.data_dir
    for scene in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, scene)):
            continue
        imgs.extend([os.path.join(data_dir, scene, '0', 'left_rgb', filename) \
            for filename in os.listdir(os.path.join(data_dir, scene, '0', 'left_rgb'))])

    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)
