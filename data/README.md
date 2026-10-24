# Data

The pipeline consumes [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
scenes; no dataset content is redistributed by this repository.

## The 173-frame evaluation set (tiny173)

Every plan_L2 number in this repository - the README results tables, the docs, and
the committed JSONs under [lite/results/](../lite/results/) - is measured on one Bench2Drive val
route:

    v1/HazardAtSideLane_Town10HD_Route373_Weather9    # 173 frames

We picked that route at random from the full Bench2Drive val routes to serve as a
small, stable, before/after quantization precision gate. The upstream converter
(`mmcv/datasets/prepare_B2D.py`, from Bench2DriveZoo) generates the train/val
metadata pkls from a train/val split JSON; we replaced the val list in that split
JSON with this single route and re-ran the converter.

Two files in this repo define the set:

| File | Size | sha256 |
|---|---|---|
| [data/splits/bench2drive_base_train_val_split_onlyone.json](splits/bench2drive_base_train_val_split_onlyone.json) | 59 B | `1240182f64899497d97d873133e77c68aaee5bf417e08da7589dbe85b1a21b62` |
| [data/infos-tiny/b2d_infos_val.pkl](infos-tiny/b2d_infos_val.pkl) (the 173-frame val pkl) | 1.4 MB | `eda635a084c3a6b66da18e31f8a5ecff900ae6e829d2009ecd05f38c6c361179` |

Two ways to obtain the eval set:

- Use the shipped pkl directly. This is the bit-exact anchor: it is the exact file
  every committed number was measured with. Place it where the config expects it
  (see below) and evaluate.
- Regenerate from Bench2Drive: prepare the Bench2Drive base data per the upstream
  [DATA_PREP](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/DATA_PREP.md),
  then put our split JSON in place of the upstream
  `data/splits/bench2drive_base_train_val_split.json` (the converter reads that
  path, no flag to change it), run the converter, and the val pkl comes out with
  the same 173 frames of the same route. Note that derived transform fields
  (`gt_boxes`, `npc2world`) can differ from the shipped pkl in the last float64
  digits between generation runs - measurement-parsing nondeterminism, verified
  harmless at plan_L2 precision - so for bit-exact comparison against our numbers,
  use the shipped pkl.

Reading the committed eval JSONs: each reports three nested subsets of the route -
`all` (173 frames), `valid_no_pollute` (163), and `fut_valid_only` (133). The
headline plan_L2 numbers quoted in the README and docs (0.690 official PyTorch,
0.704 Lite 6L@384, ...) are the `fut_valid_only` convention: frames whose future
trajectory is fully valid for the full 3-second horizon. The Lite eval script
([lite/eval/eval_planning_local.py](../lite/eval/eval_planning_local.py)) prints all three.

## Evaluation / engine verification (7B + Lite)

Follow the upstream [Bench2DriveZoo DATA_PREP](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/DATA_PREP.md)
to download the Bench2Drive base split and generate the scene metadata pkls. For
the small-route eval set used by every number in this repo, the shipped pkl above
replaces the full val pkl. Place them under the directory `ORION_B2D_DATA_ROOT`
points at (default `<ORION_LITE_PROJECT_ROOT>/data`):

    $ORION_B2D_DATA_ROOT/bench2drive/   # route folder: v1/HazardAtSideLane_Town10HD_Route373_Weather9 (+ maps)
    $ORION_B2D_DATA_ROOT/infos-tiny/b2d_infos_val.pkl
    $ORION_B2D_DATA_ROOT/infos-tiny/b2d_map_infos.pkl

This repo's own [data/](.) layout matches that default: put the route data under
`data/bench2drive/` and the pkls under [data/infos-tiny/](infos-tiny/) (both gitignored,
except the shipped 173-frame pkl). The eval config is vendored at
[configs/orion_eval_tiny173.py](../configs/orion_eval_tiny173.py); it reads `ORION_B2D_DATA_ROOT` at load time.

## Distillation data (Lite step 1)

[lite/step1_distill/run_orion_collect.py](../lite/step1_distill/run_orion_collect.py) runs the Orion teacher over Bench2Drive
scenes and saves one .npz per frame (vision features, ego status, future
trajectory, plan targets). No dataset content is committed: outputs are
gitignored, regenerable, and used only to train the student.
