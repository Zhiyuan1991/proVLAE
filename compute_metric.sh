# compute for 3dshapes
python main.py -train_seq=3 -mode=3 -z_dim=3 -checkpoint_dir=checkpoints/3dshapes_progress_z3_b8

# compute for dsprites
python main.py -train_seq=3 -mode=3 -z_dim=2 -checkpoint_dir=checkpoints/dsprite_progress_z2
