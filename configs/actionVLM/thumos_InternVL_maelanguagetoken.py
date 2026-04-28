_base_ = [
    "thumos_InternVL.py",  # model config
]
window_size = 768
scale_factor = 1
batch_size = 1
img_size = 160

gradient_accumulation_steps =2

model = dict(
    language_model=dict(
        arch_type='qwen3',
        hf_model_path=None,
        enable_debiasing=False,
        vision_only_interval=2,
        language_warmup_epochs=0,
        fusion_mode='dual_visual_cls_head',
        cls_lang_scale=1.0,
        semantic_loss_weight=0.1,
        qwen_visual_num_frames=8,
        teacher_loss_weight=0.0,
        mllm=dict(
            type="Qwen3VL_Slowfast",
            model_path="/mnt/e/OneDrive - wqa/Models/Qwen3-VL-2B-Instruct",
            quantization_vit=False,
            quantization_llm=False,
            pretrained_pth=None,
            special_tokens=['[SEG]', '<p>', '</p>', '<vp>', '</vp>'],
        ),
        tune_llm=False,
        tune_visual_encoder=False,
        tune_video_backbone=True,
        lora_llm_enable=True,
        lora_visual_enable=True,
        lora_r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_bias="none",
        torch_dtype='bf16',
        backbone_type='videomaes_language_token',
        loss_weight=0.15,
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
            dict(type="mmaction.Resize", scale=(-1, img_size * 255 // 224)),  # 182
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(img_size, img_size), keep_ratio=False),  # 160
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
            # dict(type="LoadFrames", num_clips=1, method="sliding_window", offset_frames=0),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, img_size)),
            dict(type="mmaction.CenterCrop", crop_size=img_size),
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
            # dict(type="LoadFrames", num_clips=1, method="sliding_window", offset_frames=0),
            dict(type="LoadFrames", num_clips=1, method="sliding_window"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, img_size)),
            dict(type="mmaction.CenterCrop", crop_size=img_size),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

solver = dict(
    train=dict(batch_size=batch_size, num_workers=2),
    val=dict(batch_size=batch_size, num_workers=2),
    test=dict(batch_size=batch_size, num_workers=2),
    accumulation_steps=gradient_accumulation_steps,
    dtype='float16',
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=False,
    ema=False,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    language_model=dict(
        lr=5e-6,
        weight_decay=0.01,
        custom=[
            dict(name="backbone_language_mlp", lr=1e-4, weight_decay=0.01),
            dict(name="semantic_", lr=5e-5, weight_decay=0.01),
        ],
    ),
)
scheduler = dict(type="WarmupCosineLR", warmup_epoch=5, max_epoch=120)

workflow = dict(
    logging_interval=15,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=29,
    end_epoch=120,
)

deepspeed = dict(
    base="configs/_base_/deepspeed_scripts/base.json",
    zero_stage=2,
    train_micro_batch_size_per_gpu=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    bf16=False,
    fp16=True,
    hysteresis=2,
    offload_optimizer='none',
    offload_param='none',
)

work_dir = "exps/thumos-14/actionvlm_qwen3_dualvisual"
