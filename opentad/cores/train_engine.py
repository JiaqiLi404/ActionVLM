import copy
import torch
from tqdm import tqdm
from opentad.utils.misc import AverageMeter, reduce_loss


def train_one_epoch(
        train_loader,
        model,
        optimizer,
        scheduler,
        curr_epoch,
        logger,
        model_ema=None,
        clip_grad_l2norm=-1,
        logging_interval=200,
        logging_interval_epoch=1,
        scaler=None,
        local_rank=0,
        accumulative_step=1,
        dtype=torch.float32,
        deepspeed=False
):
    """Training the model for one epoch"""
    train_iterator = enumerate(train_loader)
    if logging_interval_epoch <= 1:
        logger.info("[Train]: Epoch {:d} started".format(curr_epoch))
        if local_rank == 0:
            train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {curr_epoch}")

    losses_tracker = {}
    num_iters = len(train_loader)
    use_amp = False if scaler is None else True

    model.train()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model

    for iter_idx, data_dict in train_iterator:
        optimizer.zero_grad()

        # current learning rate
        curr_backbone_lr = None
        curr_lang_lr = None
        if hasattr(module, "backbone") and module.backbone is not None:  # if backbone exists
            if module.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
            if hasattr(module, "language_model") and module.language_model is not None:
                curr_lang_lr = scheduler.get_last_lr()[1]
        elif hasattr(module, "language_model") and module.language_model is not None:
            curr_lang_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # forward pass
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            losses = model(**data_dict, curr_epoch=curr_epoch, dtype=dtype, return_loss=True)

        if deepspeed:
            model.backward(losses["cost"]) if "cost" in losses else None
            model.step() if "cost" in losses else None
        else:
            losses["cost"] = losses["cost"] / accumulative_step  # scale the cost for accumulation
            # compute the gradients
            if use_amp and dtype == torch.float16:
                scaler.scale(losses["cost"]).backward()
            else:
                losses["cost"].backward()

            if (iter_idx + 1) % accumulative_step == 0 or (iter_idx + 1) == num_iters:
                # gradient clipping (to stabilize training if necessary)
                if clip_grad_l2norm > 0.0:
                    if use_amp and dtype == torch.float16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

                # update parameters
                if use_amp and dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # update scheduler
                scheduler.step()

                # update ema
                if model_ema is not None:
                    model_ema.update(model)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters) and (
                curr_epoch % logging_interval_epoch == 0):
            # print to terminal
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(curr_epoch, iter_idx, num_iters - 1)
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg) if "cost" in losses_tracker else "Loss=NaN"
            block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            if curr_lang_lr is not None:
                block4 = "lr_lang={:.1e}".format(curr_lang_lr) + "  " + block4
            block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))


def val_one_epoch(
        val_loader,
        model,
        logger,
        rank,
        curr_epoch,
        model_ema=None,
        use_amp=False,
        dtype=torch.float32,
):
    """Validating the model for one epoch: compute the loss"""
    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm(val_loader, disable=(rank != 0)):
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            with torch.no_grad():
                losses = model(**data_dict, curr_epoch=curr_epoch, dtype=dtype, return_loss=True)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
