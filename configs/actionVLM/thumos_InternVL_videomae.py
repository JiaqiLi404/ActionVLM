_base_ = [
    "../_base_/models/actionvlm.py",  # model config
]
annotation_path = "data/thumos-14/annotations/thumos_14_anno_cleaned.json"
class_map = "data/thumos-14/annotations/category_idx.txt"
data_path = "/mnt/e/OneDrive - wqa/Dataset/THUMOS14/rgb_videos_validation_test/"
block_list = None
feature_stride = 1
sample_stride = 4
offset_frames = 0
window_size = 768
scale_factor = 1
batch_size = 1
img_size = 160

gradient_accumulation_steps =2

model = dict(
    language_model=dict(
        tune_llm=False,
        tune_visual_encoder=False,
        tune_video_backbone=True,
        lora_llm_enable=True,
        lora_visual_enable=False,
        lora_r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_bias="none",
        torch_dtype='bf16',
        backbone_type='videomaes',
        loss_weight=0.15
    ),
    rpn_head=dict(
        projection=dict(
            max_seq_len=window_size,
        )
    )
)

dataset = dict(
    train=dict(
        type="ThumosPaddingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos-14 dataloader setting
        feature_stride=feature_stride,
        sample_stride=sample_stride,  # 1x4=4
        offset_frames=offset_frames,
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
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos-14 dataloader setting
        feature_stride=feature_stride,
        sample_stride=sample_stride,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.25,
        offset_frames=offset_frames,
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
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # thumos-14 dataloader setting
        feature_stride=feature_stride,
        sample_stride=sample_stride,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.5,
        offset_frames=offset_frames,
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
    language_model=dict(lr=5e-6, weight_decay=0.01),
)
scheduler = dict(type="WarmupCosineLR", warmup_epoch=5, max_epoch=120)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,  # set 0 to disable
    ),
    save_dict=True,
)

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

work_dir = "exps/thumos-14/actionvlm"
