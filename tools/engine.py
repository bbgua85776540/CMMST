# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""

import pprint

import mvit.models.losses as losses
import mvit.models.optimizer as optim
import mvit.utils.checkpoint as cu
import mvit.utils.distributed as du
import mvit.utils.logging as logging
import mvit.utils.metrics as metrics
import mvit.utils.misc as misc
import numpy as np
import torch
from mvit.datasets import loader
from mvit.datasets.mixup import MixUp
from mvit.models import build_model
from mvit.utils.meters import EpochTimer, TrainMeter, ValMeter
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import pickle
import numpy
from random import randint
# from ohem_loss import OhemCELoss

logger = logging.get_logger(__name__)
loss_function = nn.MSELoss()
# loss_function = nn.CrossEntropyLoss()


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
):
    """
    Perform the training for one epoch.
    Args:
        train_loader (loader): training loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        scaler (GradScaler): the `GradScaler` to help perform the steps of gradient scaling.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            mvit/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    # data_size = len(train_loader)
    # print('data_size=', data_size)
    # loss_function = nn.MSELoss()
    # loss_function = nn.L1Loss()
    # loss_function = nn.SmoothL1Loss()
    # loss_function = nn.CrossEntropyLoss()
    # loss_function = OhemCELoss(thresh=0.0105, n_min=5)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=1500, gamma=0.95)
    process_frame_nums = cfg.process_frame_nums
    count = 0
    loss_avg = 0.0

    T_start_series = time.perf_counter()
    for headmaps, salmaps, labels, series_index in train_loader:
        T_start_steps = time.perf_counter()
        model.zero_grad()
        # print('series_index = ', series_index)
        inputs, label = (headmaps.cuda(), salmaps.cuda()), labels[:, int(cfg.fps):].cuda()
        output = model(inputs)

        # print('output=', output[0].shape, 'label=', label[0].shape)
        # print('output=', output[0][0])
        # print('label=', label[0][0])
        loss = loss_function(output, label)
        loss.backward()
        for j in range(30):
            o = output[0][j].argmax()
            l = label[0][j].argmax()
            # print('output_max_', j,' =', o, '    label_max_', j,' =',l)
            # print(j, output[0][j], label[0][j])
        optimizer.step()
        T_end_steps = time.perf_counter()
        # print('loss=', loss , 'cost_time=', (T_end_steps-T_start_steps))
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        loss_avg += float(loss.cpu().detach())
        count += 1
        scheduler.step()
        # if count % 100 == 0:
        T_end_series = time.perf_counter()
        # if count%10000==0:
        #     logger.info(f"{count} loss_avg= {loss_avg / count},   lr= {learning_rate}, cost= {T_end_series - T_start_series}")
        #     T_start_series = time.perf_counter()

    logger.info(f"epoch= {cur_epoch}, count={count}  loss_avg={loss_avg / count} learning_rate={learning_rate}")
    saveModel(model, model_name='mvit-b8-pre30-lr000005-gblur49-epoch{}-ls{}.pth'.format(cur_epoch, loss_avg / count))
    return model

def saveModel(model, model_name = 'sgd-transformer.pth'):
    torch.save(model,'./result/model/' + model_name)
    print('saved model: ' + model_name)

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            mvit/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    # loss_function = nn.MSELoss()
    # loss_function = nn.L1Loss()
    # loss_function = nn.SmoothL1Loss()
    # loss_function = nn.CrossEntropyLoss()
    count = 0
    loss_avg = 0.0
    process_frame_nums = cfg.process_frame_nums
    output_user = []
    last_user = 0
    last_topic = 0
    for cur_iter, (headmaps, salmaps, labels, topic_index, user_index, series_index) in enumerate(val_loader):
        headmaps = headmaps.float()  # torch.Size([2, 4000, 120, 240])
        salmaps = salmaps.float()  # torch.Size([2, 4000, 3, 120, 240])
        # print('topic_index=', topic_index, '  user_index=', user_index)
        if last_user != user_index:  # 每个用户预测完，进行保存，并清空记录表
            pickle.dump(output_user, open('E:/work/pytorch_workplace/CMMST/result/pred_viewport/'
                        + 'ds{}_topic{}_user{}'.format(2, last_topic, last_user), 'wb'))
            last_user = int(user_index)
            last_topic = int(topic_index)
            output_user = []
        model.zero_grad()
        inputs, label = (headmaps.cuda(), salmaps.cuda()), labels[:, int(cfg.fps):].cuda()
        output = model(inputs)
        # label = label[:, -1, :]
        # o = output[0].argmax()
        # l = label[0].argmax()
        # print('output_max_index =', o, '    label_max_index =',l)
        loss = loss_function(output, label)
        # print(loss)
        loss_avg += float(loss.cpu().detach())
        ou = output.detach().cpu().reshape(output.shape[1], cfg.n_row, cfg.n_colum)
        # ou = np.array(ou)
        # argmax = np.where(ou == ou.max())
        # print(series_index, '  ==',  argmax)
        output_user.append(ou)
        count += 1
    # output_user = numpy.concatenate(output_user, axis=0)
    pickle.dump(output_user, open('E:/work/pytorch_workplace/CMMST/result/pred_viewport/'
                                  + 'ds{}_topic{}_user{}'.format(2, last_topic, last_user), 'wb'))
    logger.info(f"'eval=', 'count=', count, loss_avg={loss_avg / count}")


def train(cfg):
    """
    Train a model on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in mvit/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    if cfg.resume is not None:                     # 03 30
        state_dict = torch.load(cfg.resume).state_dict()
        model.load_state_dict(state_dict)
        print("Checkpoint {} loaded!".format(cfg.resume))

    # Create the train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        # loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
        )

        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)


def test(cfg):
    """
    Perform testing on the pretrained model.
    Args:
        cfg (CfgNode): configs. Details can be found in mvit/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(pprint.pformat(cfg))

    # Build the model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    # cu.load_test_checkpoint(cfg, model)
    if cfg.resume is not None:
        print("loading Checkpoint {}!".format(cfg.resume))
        state_dict = torch.load(cfg.resume).state_dict()
        model.load_state_dict(state_dict)
        print("Checkpoint {} loaded!".format(cfg.resume))

    # Create testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Create meters.
    test_meter = ValMeter(len(test_loader), cfg)

    eval_epoch(test_loader, model, test_meter, -1, cfg)




