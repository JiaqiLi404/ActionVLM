_base_ = [
    "thumos_InternVL_maelanguagetoken.py",
]

model = dict(
    language_model=dict(
        rerank_topk=0,
        lang_weight_mode='hybrid',
        cls_token_feature_mode='raw',
    )
)

work_dir = "exps/thumos-14/actionvlm_qwen3_hybridweight"
