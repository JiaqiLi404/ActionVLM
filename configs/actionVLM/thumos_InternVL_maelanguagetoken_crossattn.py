_base_ = [
    "thumos_InternVL_maelanguagetoken.py",
]

model = dict(
    language_model=dict(
        fusion_mode='cross_modal_cls_head',
        semantic_loss_weight=0.08,
        teacher_loss_weight=0.0,
        qwen_visual_num_frames=8,
    )
)

work_dir = "exps/thumos-14/actionvlm_qwen3_crossattn"
