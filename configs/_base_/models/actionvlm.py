base_prompt = ("This is a video reasoning for action localisation task. "
               "You should locate the start and end frames of the action in the video provided. "
               "The actions include: <actions>.")
former = dict(
    type="ActionFormer",
    projection=dict(
        type="Conv1DTransformerProj",
        in_channels=3200,
        out_channels=512,
        arch=(2, 2, 5),  # layers in embed / stem / branch
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=128,
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="ActionFormerHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

model = dict(
    type="VLLM_Detector",
    projection=None,
    language_model=dict(
        type="ActionVLM",
        mllm=dict(
            type="InternVL_Slowfast",
            model_path="ByteDance/Sa2VA-4B",
            quantization_vit=False,
            quantization_llm=False,
            pretrained_pth=None,
            special_tokens=['[SEG]', '<p>', '</p>', '<vp>', '</vp>'],
            # special_tokens=[]
        ),
        tune_llm=False,
        tune_visual_encoder=False,
        lora_llm_enable=True,
        lora_visual_enable=False,
        lora_r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_bias="none",
        torch_dtype='bf16',
        pretrained_pth=None,
        clip_window=768,
        hf_model_path='/mnt/e/OneDrive - wqa/Models/Sa2VA-4B',
        # aggregated_features=['x', 'sam2_vision_features_mean_pooling']
    ),
    base_prompt=base_prompt,
    llm_proposal_order="timestamp",
    rpn_head=former
)
