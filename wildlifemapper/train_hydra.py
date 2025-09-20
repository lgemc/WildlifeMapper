# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder - Hydra version
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

join = os.path.join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf

from dataloader_coco import build_dataset
from segment_anything import sam_model_registry
import torch.nn.functional as F
import random
from datetime import datetime
import segment_anything.utils.misc as utils
from segment_anything.network import MedSAM
from inference import evaluate, get_coco_api_from_dataset

#bowen
import train_utils

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

#bbox coordinates are provided as XYXY : left top right bottom
def show_box(box, ax):
    w, h = box[2], box[3]
    x0, y0 = box[0]-(w/2), box[1]-(h/2)
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

def create_datasets_and_loaders(cfg):
    """Create datasets and data loaders from Hydra config"""
    # Convert config to namespace for compatibility with existing code
    class Args:
        def __init__(self, cfg):
            for key, value in cfg.items():
                setattr(self, key, value)

    args = Args(cfg)

    # Create datasets
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # Optional: data sanity check (disabled by default)
    import cv2
    for step, data in enumerate(dataset_train):
        break
        image = np.transpose(np.asarray(data['image']), (1,2,0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = data['target']['boxes']
        bboxes = bboxes*1024
        print(image.shape, bboxes.shape)
        # show the example
        _, axs = plt.subplots(1, figsize=(50, 50))
        idx = random.randint(0, 7)
        axs.imshow(image)
        for bboxe in bboxes:
            show_box(bboxe.numpy(), axs)
        axs.axis("off")
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
        plt.close()
        break

    # distributed training dataset setup
    if cfg.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.custom_collate, num_workers=cfg.num_workers)
    data_loader_val = DataLoader(dataset_val, cfg.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.custom_collate, num_workers=cfg.num_workers)

    #for evaluation, coco_api
    base_ds = get_coco_api_from_dataset(dataset_val)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, base_ds, sampler_train

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Convert config to namespace for compatibility
    class Args:
        def __init__(self, cfg):
            for key, value in cfg.items():
                setattr(self, key, value)

    args = Args(cfg)

    # Set derived attributes
    args.trained_model = cfg.work_dir
    args.dist_url = 'env://'

    # bowen need to put it here since code starts before main function.
    train_utils.init_distributed_mode(args)

    # Set device
    device = torch.device(cfg.device)

    # Create datasets and loaders
    dataset_train, dataset_val, data_loader_train, data_loader_val, base_ds, sampler_train = create_datasets_and_loaders(cfg)

    # bowen - setup wandb if enabled
    if cfg.use_wandb and train_utils.is_main_process():
        import wandb

        wandb.login()
        wandb.init(
            project="aerial_detection_project",
            name=f'hydra_{cfg.model_type}',
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    seed = cfg.seed + train_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #initialize network
    sam_model, criterion, postprocessors = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint, args=args)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder, #50 objects per image
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    # bowen
    model_without_ddp = medsam_model
    if cfg.distributed:
        medsam_model = torch.nn.parallel.DistributedDataParallel(medsam_model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = medsam_model.module

    medsam_model.train()
    criterion.train()

    # img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) +
    #                         list(medsam_model.mask_decoder.parameters())
    # bowen
    mask_prompt_params =  list(model_without_ddp.mask_decoder.parameters()) \
                        + list(model_without_ddp.prompt_encoder.parameters())
    hfc_adaptor_params = list(model_without_ddp.image_encoder.hfc_embed.parameters()) \
                        + list(model_without_ddp.image_encoder.patch_embed.parameters()) \
                        + list(model_without_ddp.image_encoder.hfc_attn.parameters())
    optimizer = torch.optim.AdamW([{"params" : mask_prompt_params},
                                   {"params": hfc_adaptor_params, "lr": 0.0001}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop)

    #Number mask decoder parameters:
    train_params_list = [n for n, p in model_without_ddp.named_parameters() if p.requires_grad]
    print(train_params_list)

    # %% train
    num_epochs = cfg.num_epochs
    iter_num = 0
    best_loss = 1e10

    print("Number of training samples: ", dataset_train.__len__())

    start_epoch = 0
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("RESUMING TRAINING")
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(cfg.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            # bowen
            # medsam_model.load_state_dict(checkpoint["model"])
            model_without_ddp.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint["optimizer"])
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        #metric logging imported from DETR codebase
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10

        if cfg.distributed:
            sampler_train.set_epoch(epoch)

        for data in metric_logger.log_every(data_loader_train, print_freq, header):
            optimizer.zero_grad()
            image = data[0]
            targets = data[1]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            #passing the whole image as prompt
            b,c,h,w = image.tensors.shape
            boxes_np = np.repeat(np.array([[0,0,h,w]]), cfg.batch_size, axis=0)
            image = image.to(device)
            outputs = medsam_model(image, boxes_np)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                            for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()

            if cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(medsam_model.parameters(), cfg.clip_max_norm)
            optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            epoch_loss += losses.item()
            iter_num += 1

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        #learning rate schedular
        lr_scheduler.step()


        #Evaluation after each epoch
        test_stats, coco_evaluator = evaluate(
            medsam_model, criterion, postprocessors, data_loader_val, base_ds, device, args)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                       'epoch': epoch}
        # bowen
        if cfg.use_wandb and train_utils.is_main_process():
            wandb.log(log_stats)
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}'
        )
        ## save the latest model
        checkpoint = {
            # bowen
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if epoch%40 == 0:
            checkpoint_path = f"{cfg.work_dir}/checkpoint_epoch_{epoch}.pth"
            # bowen
            # torch.save(checkpoint, checkpoint_path)
            train_utils.save_on_master(checkpoint, checkpoint_path)

        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                # bowen
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            best_checkpoint_path = f"{cfg.work_dir}/best_checkpoint.pth"
            # bowen
            # torch.save(checkpoint, best_checkpoint_path)
            train_utils.save_on_master(checkpoint, best_checkpoint_path)

if __name__ == "__main__":
    main()