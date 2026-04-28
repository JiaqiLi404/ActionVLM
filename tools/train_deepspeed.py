import json
import os
import sys
import warnings
from dataclasses import dataclass, field

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)
import deepspeed
import torch
from mmengine.config import Config
import transformers
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)
from opentad.cores.scheduler import build_scheduler_deepspeed

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    config: str = field(
        default=None,
        metadata={"help": "path to config file"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "random seed"}
    )
    id: int = field(
        default=0,
        metadata={"help": "repeat experiment id"}
    )
    resume: str = field(
        default=None,
        metadata={"help": "resume from a checkpoint"}
    )
    not_eval: bool = field(
        default=False,
        metadata={"help": "whether not to eval, only do inference"}
    )
    disable_deterministic: bool = field(
        default=False,
        metadata={"help": "disable deterministic for faster speed"}
    )
    cfg_options: dict[str] = field(
        default=None,
        metadata={"help": "override settings"}
    )


def main():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.deepspeed is not None and args.deepspeed is None:
        args.deepspeed = cfg.deepspeed
    deepspeed_cfg = None
    zero_stage = None
    bf16 = None
    fp16 = None
    offload_optimizer = None
    offload_param = None
    gradient_accumulation_steps = None
    if isinstance(args.deepspeed, dict):
        base = args.deepspeed.get("base")
        zero_stage = args.deepspeed.get("zero_stage", 2)
        train_micro_batch_size_per_gpu = args.deepspeed.get("train_micro_batch_size_per_gpu", 1)
        gradient_accumulation_steps = args.deepspeed.get("gradient_accumulation_steps", 1)
        bf16 = args.deepspeed.get("bf16", False)
        fp16 = args.deepspeed.get("fp16", False)
        hysteresis = args.deepspeed.get("hysteresis", 2)
        offload_optimizer = args.deepspeed.get("offload_optimizer", "none")
        offload_param = args.deepspeed.get("offload_param", "none")
        deepspeed_cfg = json.load(open(base, "r"))
        deepspeed_cfg["zero_optimization"]["stage"] = zero_stage
        deepspeed_cfg["train_micro_batch_size_per_gpu"] = train_micro_batch_size_per_gpu
        deepspeed_cfg["gradient_accumulation_steps"] = gradient_accumulation_steps
        deepspeed_cfg["bf16"]['enabled'] = bf16
        deepspeed_cfg["fp16"]['enabled'] = fp16
        deepspeed_cfg["fp16"]["hysteresis"] = hysteresis
        deepspeed_cfg["zero_optimization"]["offload_param"] = {
            'device': offload_param} if offload_param != "none" else None
        deepspeed_cfg["zero_optimization"]["offload_optimizer"] = {
            'device': offload_param} if offload_param != "none" else None
    elif isinstance(args.deepspeed, str):
        deepspeed_cfg = json.load(open(args.deepspeed, "r"))
        zero_stage = deepspeed_cfg.get("zero_optimization", {}).get("stage", 2)
        bf16 = deepspeed_cfg.get("bf16", {}).get("enabled", False)
        fp16 = deepspeed_cfg.get("fp16", {}).get("enabled", False)
        offload_optimizer = deepspeed_cfg.get("zero_optimization", {}).get("offload_optimizer", None)
        offload_param = deepspeed_cfg.get("zero_optimization", {}).get("offload_param", None)
        gradient_accumulation_steps = deepspeed_cfg.get("gradient_accumulation_steps", 1)
        if offload_optimizer is not None:
            offload_optimizer = offload_optimizer.get("device", "none")
        if offload_param is not None:
            offload_param = offload_param.get("device", "none")
    zero3 = zero_stage == 3
    offload = offload_optimizer != "none" or offload_param != "none"
    dtype = torch.float32
    if bf16:
        dtype = torch.bfloat16
    if fp16:
        dtype = torch.float16

    # distributed init
    deepspeed.init_distributed()
    args.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    args.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )

    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.val,
    )

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # build model
    model = build_detector(cfg.model)

    # Model EMA
    # !!! Please notice that EMA with VLM will double the memory usage and is not recommended !!!
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model, device=f"cuda:{args.local_rank}", zero3=zero3)
    else:
        model_ema = None

    # build optimizer and scheduler
    optimizer,optim_groups = build_optimizer(cfg.optimizer, model, logger, offload=offload, return_optimizer=False)
    scheduler, max_epoch = build_scheduler_deepspeed(cfg.scheduler, len(train_loader), gradient_accumulation_steps)
    deepspeed_cfg['scheduler'] = scheduler
    deepspeed_cfg['optimizer'] = optimizer

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch")

    # deepspeed
    if deepspeed_cfg is not None:
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=optim_groups,
            config=deepspeed_cfg,
        )

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume is not None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema is not None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  # save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")
    val_loss_best = 1e6
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    fill_none_grad = cfg.solver.get("fill_none_grad", False)
    custom_save_checkpoint = cfg.solver.get("custom_save_checkpoint", False)
    best_average_mAP = 0
    for epoch in range(resume_epoch + 1, max_epoch):
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            logging_interval_epoch=cfg.workflow.get("logging_interval_epoch", 1),
            scaler=None,  # deepspeed would automatically enable fp16
            deepspeed=True,
            local_rank=args.local_rank,
            dtype=dtype,
        )


        # # save checkpoint
        # if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
        #     if args.rank == 0:
        #         save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir,
        #                         custom_save_checkpoint=custom_save_checkpoint, zero3=zero3)

        # # val for one epoch
        # if epoch >= val_start_epoch:
        #     if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
        #         val_loss = val_one_epoch(
        #             val_loader,
        #             model,
        #             logger,
        #             args.rank,
        #             epoch,
        #             model_ema=model_ema,
        #             use_amp=False,
        #             dtype=dtype,
        #             device=f"cuda:{args.local_rank}",
        #             zero3=zero3,
        #         )

        # # save the best checkpoint
        # if val_loss < val_loss_best:
        #     logger.info(f"New best epoch {epoch}")
        #     val_loss_best = val_loss
        #     if args.rank == 0:
        #         save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir,
        #                              custom_save_checkpoint=custom_save_checkpoint, zero3=zero3)

        # eval for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_eval_interval <= 0) or ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
                average_mAP = eval_one_epoch(
                    test_loader,
                    model,
                    cfg,
                    logger,
                    args.rank,
                    model_ema=model_ema,
                    use_amp=False,
                    world_size=args.world_size,
                    not_eval=args.not_eval,
                    curr_epoch=epoch,
                    dtype=dtype,
                )
                if average_mAP is not None and average_mAP > best_average_mAP:
                    best_average_mAP = average_mAP
                    logger.info(f"New best epoch {epoch}")
                    # if args.rank == 0:
                    #     save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir,map=f"_{average_mAP}",
                    #                          custom_save_checkpoint=custom_save_checkpoint, zero3=zero3)
    logger.info("Training Over...\n")
    logger.info(f"Best average mAP: {best_average_mAP}\n")


if __name__ == "__main__":
    main()
