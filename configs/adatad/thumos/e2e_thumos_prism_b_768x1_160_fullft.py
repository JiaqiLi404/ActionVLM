_base_ = ["e2e_thumos_videomae_s_768x1_160_fullft.py"]

window_size = 768
chunk_num = window_size // 16
scale_factor = 1
img_size = 288

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

model = dict(
    backbone=dict(
        backbone=dict(
            type="VideoPrismBackbone",
            model_name='videoprism_public_v1_base',

            img_size=224,
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=False,
            num_frames=8,
            drop_path_rate=0.05,

            attn_pool_num_heads=16,
            sep_pos_embed=False,
            use_flash_attn=True,
            use_fused_rmsnorm=True,
            use_fused_mlp=True,

            clip_teacher_embed_dim=1408,
            clip_teacher_final_dim=768,  # if 0, not distill final features
            clip_norm_type='l2',
            clip_return_layer=1,
            clip_student_return_interval=1,
            clip_student_return_index=None,
            clip_student_decoder='MLP_Decoder',

            use_checkpoint=True,
            checkpoint_num=12,

            use_adapter=False,
            window_size=window_size,
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            window_size=window_size,
            pretrain=None,
            data_preprocessor=False,
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
                dict(type="Rearrange", keys=["frames"], ops="b n c t h w -> b n t h w c"),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n t p c -> b t c", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) t c -> b c (t1 t)", t1=chunk_num)
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
        ),
    ),
    projection=dict(
        in_channels=768,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(lr=1e-5, weight_decay=0.05),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
    accumulation_steps=1
)

work_dir = "exps/thumos/adatad/e2e_actionformer_videomae_b_768x1_160_fullft"
