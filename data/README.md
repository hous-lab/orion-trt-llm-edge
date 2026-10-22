# Data

The pipeline consumes [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
scenes; no data is redistributed by this repository.

## Evaluation / engine verification (7B + Lite)

Follow the upstream [Bench2DriveZoo DATA_PREP](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/uniad/vad/docs/DATA_PREP.md)
to download the Bench2Drive mini split (or full split) and generate the scene
metadata pkls. Place:

    data/bench2drive/          # Bench2Drive mini split (camera + measurement files)
    data/infos-tiny/b2d_infos_val.pkl
    data/infos-tiny/b2d_map_infos.pkl

The release tree stores these under data/ (gitignored) so the config-relative
paths in run_trt_eval.sh resolve without edits.

## Distillation data (Lite step 1)

step1_distill/run_orion_collect.py runs the Orion teacher over Bench2Drive
scenes and saves one .npz per frame (vision features, ego status, future
trajectory, plan targets). No dataset content is committed: outputs are
gitignored, regenerable, and used only to train the student.
