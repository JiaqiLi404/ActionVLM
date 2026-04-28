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
window_size = 384
scale_factor = 1
batch_size = 1
img_size = 448

gradient_accumulation_steps = 2

# BaseballPitch, BasketballDunk, Billiards, CleanAndJerk, CliffDiving, CricketBowling, CricketShot, Diving,
# FrisbeeCatch, GolfSwing, HammerThrow, HighJump, JavelinThrow, LongJump, PoleVault, Shotput, SoccerPenalty,
# TennisSwing, ThrowDiscus, VolleyballSpiking
base_prompt = ("This is a video reasoning for action localisation task. "
               "You should locate the start and end frames of the action in the video provided. "
               "The actions include: <actions>.\n"
               "Here we provide the descriptions of each action:\n"
               "BaseballPitch: The individual starts by preparing and winding up with the baseball in hand, then initiates the pitching motion, and ends when the ball is released and thrown toward the batter.\n"
               "BasketballDunk: The individual starts to jump towards the hoop while holding the basketball and ends when the ball is forcefully pushed through the basket.\n"
               "Billiards: The individual starts by positioning the cue stick and aiming at the cue ball, then strikes the cue ball with the stick, and ends when the balls come to a complete stop after the shot.\n"
               "CleanAndJerk: The individual starts by gripping the barbell on the ground, then lifts it to the shoulders in the \"clean\" phase, pauses briefly, and continues by explosively lifting it overhead in the \"jerk\" phase, ending when the barbell is held steady overhead with full control.\n"
               "CliffDiving: The individual starts by standing at the edge of a cliff or platform, then jumps off and performs acrobatic movements while descending, ending when they enter the water below.\n"
               "CricketBowling: The individual starts by taking a run-up toward the wicket, then releases the ball with a straight arm towards the batsman, ending when the ball reaches the batsman or wicketkeeper.\n"
               "CricketShot: The individual starts by preparing the bat stance as the ball approaches, then swings the bat to hit the ball, ending when the ball is struck and sent away from the batsman.\n"
               "Diving: The individual starts by running or standing at the edge of a diving board or platform, then jumps and performs a controlled descent into the water, ending upon water entry.\n"
               "FrisbeeCatch: The individual starts by tracking the flying frisbee, positions their hands or body to intercept, and ends when they successfully grasp or trap the frisbee.\n"
               "GolfSwing: The individual begins by addressing the golf ball, then swings the club in a controlled arc to strike the ball, ending when the ball is launched toward the target.\n"
               "HammerThrow: The individual starts by gripping the hammer, initiates spinning rotations to build momentum, and ends when they release the hammer into the throwing sector.\n"
               "HighJump: The individual runs towards the bar, initiates the jump at the designated point, and ends upon landing on the mat after clearing the bar.\n"
               "JavelinThrow: The individual runs forward with the javelin, begins the throwing motion, and ends when the javelin is released into the air.\n"
               "LongJump: The individual begins running along the track, then takes off from the takeoff board, and ends when they land in the sandpit.\n"
               "PoleVault: The individual starts by sprinting down the runway holding the pole, plants the pole into the vault box, uses it to propel upward over the bar, and ends when they land safely on the mat.\n"
               "Shotput: The individual begins by positioning the shot near the neck, then uses a pushing motion to launch the shot forward, ending when the shot lands on the ground.\n"
               "SoccerPenalty: The individual starts by approaching the ball placed at the penalty mark and ends after striking the ball towards the goal.\n"
               "TennisSwing: The individual starts by preparing their stance and tracking the incoming ball, then swings the racket to strike the ball, ending when the ball is hit and sent back over the net.\n"
               "ThrowDiscus: The individual starts by gripping the discus, performs a spinning motion to gain momentum, and ends when the discus is released into the air.\n"
               "VolleyballSpiking: The individual begins by jumping near the net, swings their arm forcefully to hit the ball downward over the net, and ends when the ball crosses into the opponent’s court.\n"
               )

model = dict(
    base_prompt=base_prompt,
    language_model=dict(
        enable_debiasing=True,  # enable language-advantage-based debiasing
        adv_loss_weight=0.1,  # weight of the advantage regression loss
        vision_only_interval=2,  # run a vision-only epoch every N epochs to estimate the visual baseline
        use_delta_fusion=True,  # fuse language as a residual delta over visual features instead of direct replacement
        lang_residual_dropout=0.1,  # dropout on language residuals to reduce over-reliance on language
        fallback_lang_weight=0.2,  # default language weight when no valid advantage prediction is available
        lang_gate_floor=0.0,  # minimum language gate value; 0 means language can be fully suppressed
        cls_lang_scale=1.0,  # language strength for the classification branch
        loc_lang_scale=0.6,  # language strength for the localization branch; kept smaller to protect boundaries
        use_similarity_gate=True,  # additionally gate language by visual-language feature agreement
        similarity_gate_scale=8.0,  # sharpness of the similarity gate; larger means more selective gating
        tune_llm=False,
        tune_visual_encoder=True,
        lora_llm_enable=True,
        lora_visual_enable=False,
        lora_r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_bias="none",
        torch_dtype='bf16',
        backbone_type='videomaes_language_token',
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
    language_model=dict(
        lr=5e-6,
        weight_decay=0.01,
        custom=[
            dict(name="backbone_language_mlp", lr=1e-4, weight_decay=0.01),
            dict(name="lang_cls_adapter", lr=1e-4, weight_decay=0.01),
            dict(name="lang_loc_adapter", lr=1e-4, weight_decay=0.01),
            dict(name="adv_predictor", lr=1e-4, weight_decay=0.0),
        ],
    ),
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
    logging_interval=5,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=39,
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
