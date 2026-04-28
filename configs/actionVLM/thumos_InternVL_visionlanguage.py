_base_ = [
    "thumos_InternVL.py",  # model config
]
window_size = 384
scale_factor = 1
batch_size = 1

gradient_accumulation_steps =8

model = dict(
    language_model=dict(
        tune_llm=False,
        tune_visual_encoder=True,
        lora_llm_enable=True,
        lora_visual_enable=False,
        lora_r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_bias="none",
        torch_dtype='bf16',
        backbone_type='vision_language',
        loss_weight=0.1
    ),
    rpn_head=dict(
        projection=dict(
            max_seq_len=window_size,
        )
    )
)

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 512)),  # 182
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(448, 448), keep_ratio=False),  # 160
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", offset_frames=0),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 448)),
            dict(type="mmaction.CenterCrop", crop_size=448),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", offset_frames=0),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 448)),
            dict(type="mmaction.CenterCrop", crop_size=448),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

solver = dict(
    train=dict(batch_size=batch_size, num_workers=1),
    val=dict(batch_size=batch_size, num_workers=1),
    test=dict(batch_size=batch_size, num_workers=1),
    accumulation_steps=gradient_accumulation_steps,
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=False,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    language_model=dict(lr=1e-5, weight_decay=0.01),
)
scheduler = dict(type="WarmupCosineLR", warmup_epoch=5, max_epoch=100)

workflow = dict(
    logging_interval=5,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=39,
    end_epoch=100,
)

deepspeed = dict(
    base="configs/_base_/deepspeed_scripts/base.json",
    zero_stage=2,
    train_micro_batch_size_per_gpu=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=True,
    fp16=False,
    hysteresis=2,
    offload_optimizer='none',
    offload_param='none',
)

work_dir = "exps/thumos-14/actionvlm"
