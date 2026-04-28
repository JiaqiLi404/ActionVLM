_base_ = ["e2e_thumos_videomae_s_768x1_160_fullft.py"]
chunk_num = 48
window_size = 768
model = dict(
    backbone=dict(
        backbone=dict(embed_dims=768, depth=12, num_heads=12),
        # custom=dict(pretrain="pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"),
        custom=dict(pretrain="pretrained/vit_b_k710_dl_from_giant.pth",
                    post_processing_pipeline=[
                        dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                        dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                        dict(type="Interpolate", keys=["feats"], size=window_size, mode="linear"),
                    ],
                    ),
    ),
    projection=dict(in_channels=768),
)

workflow = dict(
    logging_interval=25,
    checkpoint_interval=5,
    val_loss_interval=-1,
    # val_eval_interval=5,
    # val_start_epoch=39,
    end_epoch=60,

    val_eval_interval=5,
    val_start_epoch=39,
)

work_dir = "exps/thumos/adatad/e2e_actionformer_videomae_b_768x1_160_fullft"
