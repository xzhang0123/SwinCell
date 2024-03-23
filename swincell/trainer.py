import os
import pdb
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from swincell.utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            # print(logits.shape,target.shape)
            if args.cellpose:
                loss_func1 = loss_func[0]
                loss_func2 = loss_func[1]
                #  weight_factor* flow loss     +      cell probability loss, weight_factor set to 5 according to paper
                loss = 5*loss_func1(logits[:,1:], target[:,1:]) + loss_func2(logits[:,0], target[:,0])

            else:
                # print(logits.shape,target.shape)
                loss = loss_func(logits[:,0], target[:,0])
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    if args.save_temp_img:
        img_raw = data[0,0,:,:,:].detach().cpu().numpy()
        img_raw = (img_raw - np.min(img_raw)) / (np.max(img_raw) - np.min(img_raw))
        img_raw  =np.max(img_raw,axis=-1)*255
        img_raw = img_raw.astype(np.uint8)
        #dimenstion is (batch_size, channel, height, width) 
        img_gt= target[0,0,:,:,:].detach().cpu().numpy()
        # img_gt= (img_gt - np.min(img_gt)) / (np.max(img_gt) - np.min(img_gt))
        img_gt =np.max(img_gt,axis=-1)*255
        img_gt= img_gt.astype(np.uint8)

        img_pred = logits[0][0,:,:,:].detach().cpu().numpy()
        img_pred = (img_pred  - np.min(img_pred)) / (np.max(img_pred) - np.min(img_pred))
        img_pred  =np.max(img_pred,axis=-1)*255
        img_pred = img_pred.astype(np.uint8)
        img_list = [img_raw, img_gt, img_pred]
    else:
        img_list = None
    return run_loss.avg, img_list


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()   # cell probs
    # run_acc2 = AverageMeter()  # flows

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits) # a list of length 4 (number of input channels =4)
            # y_pred binarized
            # val_output_convert = [post_pred(post_sigmoid(val_pred_tensor[0])) for val_pred_tensor in val_outputs_list]

            # cell probs channel
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor[0])) for val_pred_tensor in val_outputs_list]
            val_label_convert = [val_label_tensor[0] for val_label_tensor in val_labels_list]

            print(len(val_label_convert),len(val_output_convert),val_label_convert[0].shape,val_output_convert[0].shape)
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_label_convert)
            #validate with the binary masks 
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                Dice_Class = run_acc.avg[0]

                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    ", Cell dice class1:",
                    Dice_Class,

                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
        if args.save_temp_img:
            img_raw = data[0,0,:,:,:].detach().cpu().numpy()
            # print(img_raw.shape)
            img_raw = (img_raw - np.min(img_raw)) / (np.max(img_raw) - np.min(img_raw))
            img_raw  =np.max(img_raw,axis=-1)*255
            img_raw = img_raw.astype(np.uint8)   
            img_gt= target[0,0,:,:,:].detach().cpu().numpy()
            # print(img_gt.shape)
            img_gt =np.max(img_gt,axis=-1)*255
            img_gt= img_gt.astype(np.uint8)
            # print(val_output_convert[0].shape)
            # img_pred = val_output_convert[0][0,:,:,:].detach().cpu().numpy()
            img_pred = val_output_convert[0][:,:,:].detach().cpu().numpy() #modified for flows

            img_pred = (img_pred  - np.min(img_pred)) / (np.max(img_pred) - np.min(img_pred))
            img_pred  =np.max(img_pred,axis=-1)*255
            img_pred = img_pred.astype(np.uint8)
            img_list = [img_raw, img_gt, img_pred]
        else:
            img_list = None

    return run_acc.avg, img_list


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss, train_img_list = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc,val_img_list = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            if args.rank == 0:

                print(
                    "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
                    ", Dice_Class1:",
                    val_acc[0],
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )
                print(epoch, val_acc)
                if writer is not None:
                    writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
                    if args.save_temp_img:
                        writer.add_image("Validation/x1_raw", val_img_list[0], epoch, dataformats="HW")
                        writer.add_image("Validation/x1_gt", val_img_list[1], epoch, dataformats="HW")
                        writer.add_image("Validation/x1_prediction", val_img_list[2], epoch, dataformats="HW")

                        writer.add_image("Training/x1_raw", train_img_list[0], epoch, dataformats="HW")
                        writer.add_image("Training/x1_gt", train_img_list[1], epoch, dataformats="HW")
                        writer.add_image("Training/x1_prediction", train_img_list[2], epoch, dataformats="HW")
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
                val_avg_acc = np.mean(val_acc)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("new best model, saving model.pt")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
