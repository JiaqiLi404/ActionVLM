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
        type="ActionReasoner",
        videomae_version='iv_2_5',
        mllm_hf_name_or_path='/mnt/e/OneDrive - wqa/Models/LLaVA-ST-Qwen2-7B',
        mllm_lora_path=None,
        mllm_embedding_dim=768,
        clip_window=768,
    ),
    # pipeline=['language_model', 'rpn_head', 'language_model'],
pipeline=['language_model', 'rpn_head'],
    base_prompt=base_prompt,
    llm_proposal_order="timestamp",
    rpn_head=former
)
