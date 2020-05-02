# visualize results for 3dshapes
python main.py -train_seq=2 -mode=2 -z_dim=3 -checkpoint_dir=checkpoints/3dshapes_progress_z3_b8

# visualize results for dsprites
python main.py -train_seq=3 -mode=2 -z_dim=2 -checkpoint_dir=checkpoints/dsprite_progress_z2

# visualize results for celebA
python main.py -train_seq=4 -mode=2 -z_dim=7 -checkpoint_dir=checkpoints/Celeb_progress_z7_exp1
