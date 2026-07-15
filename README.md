# GADM: Geometry-Aware Diffusion Matching

<!-- TODO(release): replace with the exact paper title, author list, and paper link -->
Official implementation of the ICRA 2026 submission
**"GADM: Geometry-Aware Diffusion Matching for Multi-Robot Relative Pose Estimation"** — *Suyoung Kang et al.*

<!-- TODO(release): add teaser/pipeline figure, e.g. docs/assets/teaser.png -->

GADM is a geometry-regularized, diffusion-based image matching method for
two-view relative pose estimation. Diffusion-based matchers learn the
assignment matrix purely from correspondence supervision; GADM additionally
constrains the learned assignments with **epipolar geometry**:

- **Sampson epipolar regularization.** During training, the predicted
  correspondences are scored against the ground-truth relative pose with a
  differentiable Sampson epipolar error, and this geometric residual is added
  to the matching objective
  (`GADM/scripts/models/utils/pose_utils.py`, `GADM/scripts/models/matchers/gadm.py`):

  `L = w_match · L_match + L_confidence + w_epi · L_sampson`

  Matches that are photometrically plausible but geometrically inconsistent are
  penalized, steering the diffusion matcher toward pose-consistent assignments.
- **Pose-aware evaluation.** The evaluation suite reports relative pose AUC with
  both robust (RANSAC) and non-robust (DLT) estimators, plus an
  occlusion-robustness protocol (`GADM/generate_occlusion_data.py`).
- **Regularization-strength study.** Ready-made configs sweep the epipolar
  weight (`configs/superpoint+gadm_megadepth_epi1e2.yaml`, `configs/hyperparam/`).

GADM builds upon the excellent [DiffGlue](https://github.com/SuhZhang/DiffGlue)
codebase (Zhang & Ma, ACM MM 2024), which follows the
[glue-factory](https://github.com/cvg/glue-factory) training framework. We are
grateful to the authors for releasing their code — see
[Acknowledgements](#license--acknowledgements) for what is inherited and what is new.

> **Naming note.** The GADM matcher lives in `scripts/models/matchers/gadm.py`;
> `matchers.diffglue` is kept as a backward-compatible alias so that older
> configs and checkpoints still load. The conda environment is named `ggdm`
> (GADM's working title during development).

## Installation

Python 3.9 with PyTorch (CUDA) is required:

```bash
git clone https://github.com/suyoung1222/GADM.git
cd GADM
conda env create -f environment.yml   # creates the "ggdm" conda env
conda activate ggdm
```

or install the pinned pip dependencies into your own environment:

```bash
pip install -r requirements.txt
```

## Pretrained Weights

| File | Used for | Source |
|---|---|---|
| GADM checkpoint | **GADM (ours)** — evaluation & inference | <!-- TODO(release): Hugging Face / Google Drive link --> *link coming soon* |
| `SP_DiffGlue.tar` | DiffGlue baseline & demo | [DiffGlue release](https://drive.google.com/drive/folders/1YHd7MJaKki7e5wHqepXJLVboGYxmyRf2?usp=sharing) → place in `demo/models/weights/` |
| `superpoint_v1.pth` | SuperPoint extractor | auto-downloaded from [magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) *(non-commercial research license)* |

## Data Preparation

Benchmark datasets (MegaDepth-1500, HPatches) are downloaded automatically on
first use by the evaluation scripts. Dataset, checkpoint, and result locations
default to `GADM/data/`, `GADM/outputs/training/`, and `GADM/outputs/results/`,
and can be redirected with the environment variables `GADM_DATA_PATH`,
`GADM_TRAINING_PATH`, and `GADM_EVAL_PATH` (see `GADM/scripts/settings.py`).

For MegaDepth training, follow the
[upstream DiffGlue instructions](docs/README_DiffGlue_upstream.md) to obtain the
MegaDepth dataset (~420 GB). Optionally cache SuperPoint features to speed up
training:

```bash
cd GADM
python -m scripts.cache.export_megadepth --method sp --num_workers 8
```

## Training

GADM follows the two-stage schedule of DiffGlue/LightGlue. Stage 1 (synthetic
homography pre-training on Oxford-Paris) is unchanged from upstream:

```bash
cd GADM
python -m scripts.train SP+DiffGlue_homography \
    --conf scripts/configs/superpoint+diffglue_homography.yaml --run_benchmarks
```

Stage 2 fine-tunes on MegaDepth with GADM's Sampson epipolar regularization
(`epi_weight: 1.0`, `matcher_loss_weight: 0.1`):

```bash
python -m scripts.train SP+GADM_megadepth \
    --conf scripts/configs/superpoint+gadm_megadepth.yaml \
    train.load_experiment=SP+DiffGlue_homography --distributed
```

To study the regularization strength, use
`scripts/configs/superpoint+gadm_megadepth_epi1e2.yaml` (`epi_weight: 1e2`) or
the sweep configs in `scripts/configs/hyperparam/`.

## Evaluation

Relative pose estimation on MegaDepth-1500 (reports pose AUC @5°/10°/20° with
both RANSAC and DLT estimators):

```bash
cd GADM
python -m scripts.eval.megadepth1500 --conf superpoint+gadm-official \
    --checkpoint /path/to/gadm_checkpoint.tar
```

Homography estimation on HPatches:

```bash
python -m scripts.eval.hpatches --conf superpoint+gadm-official \
    --checkpoint /path/to/gadm_checkpoint.tar
```

The DiffGlue baseline is evaluated the same way with
`--conf superpoint+diffglue-official --checkpoint ../demo/models/weights/SP_DiffGlue.tar`.

**Occlusion robustness.** To reproduce the occlusion experiments, generate the
occluded pair lists and images with `GADM/generate_occlusion_data.py` and
`demo/images/generate_occlusion.py`, then point the evaluation to the generated
`pairs_calibrated_occ*.txt` files (`data.pairs=...`).

**Demo.** A quick image-matching / relative-pose demo with the DiffGlue weights:

```bash
cd demo && python demo.py
python demo_rel_pose_estimation.py
```

## Repository Structure

```
GADM/
  scripts/
    train.py                        # two-stage training loop (glue-factory style)
    models/
      matchers/gadm.py              # ★ GADM matcher: Sampson-regularized diffusion matching loss
      matchers/diffglue.py          # backward-compatibility alias for older configs/checkpoints
      utils/pose_utils.py           # ★ Sampson epipolar loss, epipolar geometry, PnP utilities
      diffusers/                    # diffusion process (DDPM/DDIM)
      extractors/superpoint.py      # SuperPoint (Magic Leap, non-commercial — see NOTICE.md)
    configs/
      superpoint+gadm_megadepth.yaml       # ★ main GADM training config (epi_weight 1.0)
      superpoint+gadm_megadepth_epi1e2.yaml# ★ strong-regularization variant
      superpoint+gadm-official.yaml        # ★ GADM evaluation config
      hyperparam/                          # ★ epipolar-weight sweep configs
    eval/megadepth1500.py           # relative pose eval (RANSAC + DLT AUC ★)
    eval/hpatches.py                # homography eval
  generate_occlusion_data.py        # ★ occlusion-robustness experiment data
demo/                               # upstream DiffGlue demo + relative-pose demos ★
docs/                               # upstream DiffGlue README (provenance)
```

★ = components introduced or extended by GADM.

## Real-Robot & Gazebo Experiments (ROS1/ROS2) — Coming Soon

The multi-robot relative pose estimation experiments from the paper (LIMO
robots, Gazebo simulation, ROS1/ROS2 nodes) will be released in a companion
repository. Stay tuned.

## License & Acknowledgements

This repository is released under the [MIT License](LICENSE).

It is an extension of [DiffGlue](https://github.com/SuhZhang/DiffGlue)
(MIT, © 2024 SuhZhang): the diffusion matching backbone, the training framework
(derived from [glue-factory](https://github.com/cvg/glue-factory)), and the
evaluation harness originate there. The Sampson epipolar regularization, pose
utilities, DLT-based pose evaluation, occlusion-robustness protocol, and the
regularization-strength study are new in GADM.

Third-party components keep their own licenses:

- **SuperPoint** (`GADM/scripts/models/extractors/superpoint.py`,
  `demo/models/superpoint.py`) — © Magic Leap, Inc., **academic / non-commercial
  research use only**; weights are auto-downloaded from the official repository
  (see [NOTICE](GADM/scripts/models/extractors/NOTICE.md)).
- **DDPM / DDIM** diffusion utilities — from the respective official releases,
  via DiffGlue.

## Citation

If you find this work useful, please cite GADM and DiffGlue:

<!-- TODO(release): replace with the final ICRA 2026 BibTeX once available -->
```bibtex
@inproceedings{kang2026gadm,
  title={GADM: Geometry-Aware Diffusion Matching for Multi-Robot Relative Pose Estimation},
  author={Kang, Suyoung and others},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026},
  note={Under review}
}

@inproceedings{zhang2024diffglue,
  title={DiffGlue: Diffusion-Aided Image Feature Matching},
  author={Zhang, Shihua and Ma, Jiayi},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  pages={8451--8460},
  year={2024}
}
```
