from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import adabound

hyp = {'lr0': 0.001,  # initial learning rate
       'lrf': -5.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=8, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-1cls-tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/senior2bbb_depth-post_0_0.85.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--rescale_every_n_batches", default=16, help="when to rescale images for multi-scale training")
    parser.add_argument("--unfreeze_at_epoch", type=int, default=1, help="epoch number to unfreeze loaded pretrained weights: "
                                                               "never (-1), first epoch (0), or after some epochs (> 0).")
    parser.add_argument("--checkpoints", type=str, default="checkpoints/", help="directory where to save checkpoints")
    parser.add_argument("--output", type=str, default="output/", help="directory where to save output")
    parser.add_argument("--metric_for_best", type=str, default="val_mAP", help="criterion for choosing best model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(f"{opt.output}", exist_ok=True)
    os.makedirs(f"{opt.checkpoints}", exist_ok=True)
    results_file = os.path.join(opt.output, 'results.txt')

    # Get data configuration
    data_configs = [parse_data_config(cfg_file) for cfg_file in opt.data_config.split(',')]
    # train_path = data_config["train"]
    # class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    # optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=0.1)

    # If specified we start from checkpoint
    st_epoch = 0
    clean_results = True
    if opt.pretrained_weights:
        weights = [w for w in opt.pretrained_weights.split(',')]
        if len(weights) == 1 and weights[0].endswith('.pth'):
            chkpt = torch.load(weights[0], map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])
            st_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
            del chkpt
            clean_results = False
        else:
            for k, w in enumerate(weights):
                _ = model.load_darknet_weights(w, k=k, cutoff=-1)
                # Remove old results

    if clean_results and not opt.nosave:
        debug_images = os.path.join(opt.output, '*_batch*.jpg')
        for f in glob.glob(debug_images) + glob.glob(results_file):
            os.remove(f)

    datasets = [ListDataset(cfg["train"], img_norm=cfg['normalization'], color_map=cfg['color_map'])
                for cfg in data_configs]

    # Get dataloader
    multidataset = ParallelDataset(datasets,
                                   augment=True,
                                   multiscale=opt.multiscale_training,
                                   rescale_every_n_batches=opt.rescale_every_n_batches)

    # train_dict = dict()
    # nc = -1
    # for i in range(len(data_cfg)):
    #     train_cfg = parse_data_cfg(data_cfg[i])
    #     nc_i = int(train_cfg['classes'])  # number of classes
    #
    #     if nc > 0:
    #         assert nc_i == nc
    #     nc = nc_i
    #
    #     train_dict[data_cfg[i]] = dict(
    #         data=train_cfg['train'],
    #         normalization=train_cfg['normalization'] if 'normalization' in train_cfg else None,
    #         apply_cmap=int(train_cfg['apply_cmap']) if 'apply_cmap' in train_cfg else False
    #     )

    dataloader = torch.utils.data.DataLoader(
        multidataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=multidataset.collate_fn,
    )

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in (1./3, 2./3)], gamma=0.5)
    scheduler.last_epoch = st_epoch - 1

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    nb = len(dataloader)
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches

    for epoch in range(st_epoch, st_epoch + opt.epochs):

        model.train()

        if epoch == opt.unfreeze_at_epoch:
            for name, p in model.named_parameters():
                p.requires_grad = True

        start_time = time.time()
        # loss_bat  ches_tr = []
        mloss = torch.zeros(5).to(device)
        print(('\n%8s%12s' + '%10s' * 7) %
              ('Epoch', 'mem', 'xy', 'wh', 'conf', 'cls', 'tr_loss', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for batch_i, (batch_data, img_size) in pbar:
            batches_done = len(dataloader) * epoch + batch_i

            paths, imgs, targets = list(zip(*batch_data))

            imgs = [Variable(x.to(device)) for x in imgs]
            targets = Variable(targets[0].to(device), requires_grad=False)

            # Plot images with bounding boxes
            if epoch == 0 and batch_i == 0 and not opt.nosave:
                for k, imgs_k in enumerate(imgs):
                    plot_images(imgs=imgs_k, targets=targets,
                                fname=os.path.join(opt.output, 'train_batch-%g.%g.jpg') % (batch_i, k))

            # SGD burn-in
            # if epoch == 0 and batch_i <= n_burnin:
            #     lr = hyp['lr0'] * (batch_i / n_burnin) ** 4
            #     for pg in optimizer.param_groups:
            #         pg['lr'] = lr

            loss, loss_items, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            # log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            # metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            #
            # # Log metrics at each YOLO layer
            # for i, metric in enumerate(metrics):
            #     formats = {m: "%.6f" for m in metrics}
            #     formats["grid_size"] = "%2d"
            #     formats["cls_acc"] = "%.2f%%"
            #     row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            #     metric_table += [[metric, *row_metrics]]
            #
            #     # Tensorboard logging
            #     tensorboard_log = []
            #     for j, yolo in enumerate(model.yolo_layers):
            #         for name, metric in yolo.metrics.items():
            #             if name != "grid_size":
            #                 tensorboard_log += [(f"{name}_{j+1}", metric)]
            #     tensorboard_log += [("loss", loss.item())]
            #     logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # log_str += AsciiTable(metric_table).table
            # log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            # log_str += f"\n---- ETA {time_left}"

            # print(log_str)

            model.seen += imgs[0].size(0)
            # loss_batches_tr += [loss.item()]

            mloss = (mloss * batch_i + loss_items) / (batch_i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            s = ('%10s' * 2 + '%10.3g' * 7) % (
                '%g/%g' % (epoch, st_epoch + opt.epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

        precision = recall = AP = f1 = val_loss = 0

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            eval_losses, results = evaluate(
                model,
                data_configs,
                opt.output,
                # path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
                num_workers=8
            )

            metric_names = ["val_precision", "val_recall", "val_mAP", "val_f1", "val_loss"]
            eval_metrics = {m: r.mean() for m, r in zip(metric_names, results[:-1] + (eval_losses,))}

            # logger.list_of_scalars_summary(eval_metrics, epoch)

            ap_class = results[-1]  # in 1-class problem: ap_class[0] == val_mAP

            # summary_table = [['#epoch'] + [metric_name for metric_name, _ in evaluation_metrics]]
            # summary_table += [['%d/%d' % (epoch+1, opt.epochs)] + ["%.5f" % metric_val for _, metric_val in evaluation_metrics]]
            # print(AsciiTable(summary_table).table)

            # Print class APs and mAP
            # ap_table = [["Index", "Class name", "AP"]]
            # for i, c in enumerate(ap_class):
            #     ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            # print(AsciiTable(ap_table).table)
            # print(f"---- mAP {AP.mean()}")

            # s = ('%8s%12s' + '%10.3g' * 7) % (
            #     '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, len(targets), img_size)

            # Write epoch results
            if not opt.nosave:
                with open(results_file, 'a') as file:
                    file.write(s + '%11.3g' * 5 % tuple([eval_metrics[m] for m in metric_names]) + '\n')

        # Save training results
        save = (not opt.nosave) or (epoch == opt.epochs - 1)
        if save:
            chkpt = {'epoch': epoch,
                     'eval_metrics': eval_metrics,
                     'metric_for_best': opt.metric_for_best,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # latest
            torch.save(chkpt, f"{opt.checkpoints}/yolov3_ckpt_latest.pth")

            # best, based on mAP
            it_is_best = True  # assume we got some metric's best value up to this point
            if os.path.exists(f"{opt.checkpoints}/yolov3_ckpt_best.pth"):
                chkpt_best = torch.load(f"{opt.checkpoints}/yolov3_ckpt_best.pth")
                if 'metric_for_best' in chkpt_best:
                    assert chkpt_best['metric_for_best'] == opt.metric_for_best
                    mval_best = chkpt_best['eval_metrics'][opt.metric_for_best]
                    if mval_best >= eval_metrics[opt.metric_for_best]:  # in case current is not better than best
                        it_is_best = False

            if it_is_best:
                torch.save(chkpt, f"{opt.checkpoints}/yolov3_ckpt_best.pth")

            # periodic saving, every opt.checkpoint_interval
            if opt.checkpoint_interval and epoch % opt.checkpoint_interval == 0:
                torch.save(chkpt, f"{opt.checkpoints}/yolov3_ckpt_%d.pth" % epoch)

            # Delete checkpoint
            del chkpt

        scheduler.step()


