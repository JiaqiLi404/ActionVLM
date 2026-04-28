# Towards Mitigating Modality Bias in Vision-Language Models for Temporal Action Localization

Temporal Action Localization (TAL) requires identifying both the boundaries and categories of actions in untrimmed videos. While vision-language models (VLMs) offer rich semantics to complement visual evidence, existing approaches tend to overemphasize linguistic priors at the expense of visual performance, leading to a pronounced modality bias. We propose ActionVLM, a vision-language aggregation framework that systematically mitigates modality bias in TAL. Our key insight is to preserve vision as the dominant signal while adaptively exploiting language only when beneficial. To this end, we introduce (i) a debiasing reweighting module that estimates the language advantage-the incremental benefit of language over vision-only predictions-and dynamically reweights language modality accordingly, and (ii) a residual aggregation strategy that treats language as a complementary refinement rather than the primary driver.

## TODOs
- [x] First version code release
- [x] Supporting deepspeed training
- [ ] Sorting the final version code
- [ ] Release the model checkpoints

## Notes
This repo is based on [OpenTAD](https://github.com/sming256/OpenTAD). 
In this first version, we implement Qwen3-VL as the backbone and supress the pure vision-only model by 0.5% mAP in Thumos-14 when both using the VideoMAEv2 vision backbone, and supress the VideoMAEv1 variants by ~3%. In the final version, we further address the modality bias. It will be released once 1. the adaptation to Qwen-series is done and 2. the redundant code and branches are cleaned up.

## 🖊️ Citation

If you think this repo is helpful, please cite us:

```bibtex
@inproceedings{li2026towards,
  title={Towards Mitigating Modality Bias in Vision-Language Models for Temporal Action Localization},
  author={Li, Jiaqi and Wang, Guangming and Zheng, Shuntian and Ni, Minzhe and Lu, Xiaoman and Ye, Guanghui and Guan, Yu},
  journal={Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```

If you have any questions, please contact: `li1962279338@gmail.com`.
