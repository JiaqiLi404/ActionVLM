import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

from mmengine.dataset import Compose
from mmengine.registry import MODELS as MM_BACKBONES
from einops import rearrange, reduce

BACKBONES = MM_BACKBONES


class BackboneWrapper(nn.Module):
    def __init__(self, cfg):
        super(BackboneWrapper, self).__init__()
        custom_cfg = cfg.custom
        model_cfg = copy.deepcopy(cfg)
        model_cfg.pop("custom")

        # build the backbone
        self.model = BACKBONES.build(model_cfg)

        # custom settings: pretrained checkpoint, post_processing_pipeline, norm_eval, freeze_backbone
        # 1. load the pretrained model
        self.backbone_type = 'ViT'
        if hasattr(custom_cfg, "pretrain") and custom_cfg.pretrain is not None:
            state_dict = torch.load(custom_cfg.pretrain)
            state_dict = state_dict['module'] if 'module' in state_dict else state_dict
            if "InternVideo2" in custom_cfg.pretrain:
                state_clip = torch.load("pretrained/InternVideo2_clip.bin")
                state_clip = state_clip['module'] if 'module' in state_clip else state_clip
                for key in list(state_clip.keys()):
                    if "clip_decoder" in key or 'clip_pos_embed' in key:
                        state_dict[key] = state_clip[key]

            for key in list(state_dict.keys()):
                if not key.startswith('backbone'):
                    new_key = 'backbone.' + key
                    state_dict[new_key] = state_dict.pop(key)
                    key = new_key
                if "k710_dl_from_giant.pth" in custom_cfg.pretrain:
                    if "fc1" in key:
                        new_key = key.replace("fc1", "layers.0.0")
                        state_dict[new_key] = state_dict.pop(key)
                    if "fc2" in key:
                        new_key = key.replace("fc2", "layers.1")
                        state_dict[new_key] = state_dict.pop(key)
                    if "patch_embed.proj." in key:
                        new_key = key.replace("patch_embed.proj.", "patch_embed.projection.")
                        state_dict[new_key] = state_dict.pop(key)
            res = self.model.load_state_dict(state_dict, strict=False)
            print("Missing keys for Backbone:", res.missing_keys)
            print("Unexpected keys for Backbone:", res.unexpected_keys)
            # load_checkpoint(self.model, custom_cfg.pretrain, map_location="cpu")

            if "InternVideo2" in custom_cfg.pretrain:
                # self.model.backbone = self.model.backbone
                self.backbone_type = 'InternVideo2'
                self.window_size = custom_cfg.window_size
        else:
            print(
                "Warning: no pretrain path is provided, the backbone will be randomly initialized, "
                "unless you have initialized the weights in the model.py."
            )

        # 2. pre_processing_pipeline
        self.need_data_preprocessor = custom_cfg.data_preprocessor if hasattr(custom_cfg, "data_preprocessor") else True
        if hasattr(custom_cfg, "pre_processing_pipeline"):
            self.pre_processing_pipeline = Compose(custom_cfg.pre_processing_pipeline)
        else:
            self.pre_processing_pipeline = None

        # 3. post_processing_pipeline for pooling and other operations
        if hasattr(custom_cfg, "post_processing_pipeline"):
            self.post_processing_pipeline = Compose(custom_cfg.post_processing_pipeline)
        else:
            self.post_processing_pipeline = None

        # 4. norm_eval: set all norm layers to eval mode
        self.norm_eval = getattr(custom_cfg, "norm_eval", True)

        # 5. freeze_backbone: whether to freeze the backbone, default is False
        self.freeze_backbone = getattr(custom_cfg, "freeze_backbone", False)

        print("freeze_backbone: {}, norm_eval: {}".format(self.freeze_backbone, self.norm_eval))

        # 6. whether to use temporal activation checkpointing
        self.use_temporal_checkpointing = getattr(custom_cfg, "temporal_checkpointing", False)
        if self.use_temporal_checkpointing:
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_num"
            ), "temporal_checkpointing_chunk_num should be provided when using temporal checkpointing"
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_dim"
            ), "temporal_checkpointing_chunk_dim should be provided when using temporal checkpointing"
            self.temporal_checkpointing_chunk_num = custom_cfg.temporal_checkpointing_chunk_num
            self.temporal_checkpointing_chunk_dim = custom_cfg.temporal_checkpointing_chunk_dim

    def forward(self, frames, masks=None):
        # two types: snippet or frame

        # snippet: 3D backbone, [bs, T, 3, clip_len, H, W]
        # frame: 3D backbone, [bs, 1, 3, T, H, W]

        # set all normalization layers
        self.set_norm_layer()

        # data preprocessing: normalize mean and std
        if self.need_data_preprocessor:
            frames, _ = self.model.data_preprocessor.preprocess(
                self.tensor_to_list(frames),  # need list input
                data_samples=None,
                training=False,  # for blending, which is not used in openTAD
            )

        # pre_processing_pipeline:
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]

        # go through the video backbone
        if self.freeze_backbone:  # freeze everything even in training
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    features = self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                    )
                else:
                    features = self.model.backbone(frames)

        else:  # let the model.train() or model.eval() decide whether to freeze
            if self.use_temporal_checkpointing:
                features = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                )
            else:
                if self.backbone_type == 'ViT':
                    features = self.model.backbone(frames)
                elif self.backbone_type == 'InternVideo2':
                    stride = 1
                    frames = frames[::stride]
                    batches = frames.shape[0]
                    N = self.window_size // 8 // stride
                    B = batches // N
                    features = self.model.backbone(frames)
                    features = features[:, 1:, :]
                    # features = features[0].transpose(1, 0)
                    # features = features[:, -1, 1:, :]
                    # features = features.squeeze(1)
                    # step 2: reshape to [B*N, 8, 256, C]
                    x = rearrange(features, 'bn (t s) c -> bn t s c', t=8, s=144)
                    # step 3: spatial pool → [B*N, 8, C]
                    x = reduce(x, 'bn t s c -> bn t c', 'mean')
                    # step 4: merge snippets → [B, N*8, C]
                    x = rearrange(x, '(b n) t c -> b c (n t)', b=B, n=N)
                    x = F.interpolate(x, size=self.window_size, mode="nearest")
                    x = x.to(torch.float32)
                    return x

        # unflatten and pool the features
        if isinstance(features, (tuple, list)):
            features = torch.cat([self.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1)
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)

        # # apply mask
        # if masks is not None and features.dim() == 3:
        #     features = features * masks.unsqueeze(1).detach().float()

        # make sure detector has the float32 input
        features = features.to(torch.float32)
        return features

    def tensor_to_list(self, tensor):
        return [t for t in tensor]

    def unflatten_and_pool_features(self, features, batches, num_segs):
        # unflatten the batch dimension and num_segs dimension
        features = features.unflatten(dim=0, sizes=(batches, num_segs))  # [bs, num_seg, ...]

        # convert the feature to [B,C,T]: pooling and other operations
        if self.post_processing_pipeline is not None:
            features = self.post_processing_pipeline(dict(feats=features))["feats"]
        return features

    def set_norm_layer(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                    m.eval()

                    for param in m.parameters():
                        param.requires_grad = False

    def temporal_checkpointing(self, frames, chunk_num, chunk_dim):
        """Temporal Checkpointing for Video Backbone.

        Temporal checkpointing will 1) split the video frames along the temporal dimension and sequentially forward each chunk with
        no gradients. 2) The backward pass will recompute the intermediate activations and compute each chunk's gradient. 3) Backbone's
        gradients will be accumulated along different chunks.

        Args:
            frames (Tensor): input frames, [B*N,3,T,H,W]
            chunk_num (int): number of chunks to split the temporal dimension
            chunk_dim (int): input shape is [B*N,3,T,H,W], so either dim=0 or 2 is fine
        """

        def _inner_forward(frames):
            return self.model.backbone(frames)

        video_feat = []
        for mini_frames in torch.chunk(frames, chunk_num, dim=chunk_dim):  # B*N is chunked
            # we can use torch.cp.checkpoint to implement an efficient temporal checkpointing mechanism
            mini_feat = cp.checkpoint(
                _inner_forward,
                mini_frames,
                use_reentrant=False,
            )
            video_feat.append(mini_feat)

        if isinstance(video_feat[0], (tuple, list)):
            video_feat = [torch.cat([f[idx] for f in video_feat], dim=chunk_dim) for idx in range(len(video_feat[0]))]
        else:
            video_feat = torch.cat(video_feat, dim=chunk_dim)
        return video_feat
