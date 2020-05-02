# training for 3dshapes
python main.py -train_seq=1 -mode=1 -z_dim=3 -beta=8 -KL=True -coff=0.5 -epoch_size=15 -batch_size=100 -checkpoint_dir=checkpoints/3dshapes_progress_z3_b8
python main.py -train_seq=2 -mode=1 -z_dim=3 -beta=8 -KL=True -coff=0.5 -epoch_size=15 -batch_size=100 -checkpoint_dir=checkpoints/3dshapes_progress_z3_b8

# training for dsprites
python main.py -train_seq=1 -mode=1 -z_dim=2 -beta=1 -KL=True -fadein=True -coff=0.5 -epoch_size=30 -batch_size=128 -checkpoint_dir=checkpoints/dsprite_progress_z2
python main.py -train_seq=2 -mode=1 -z_dim=2 -beta=1 -KL=True -fadein=True -coff=0.5 -epoch_size=30 -batch_size=128 -checkpoint_dir=checkpoints/dsprite_progress_z2
python main.py -train_seq=3 -mode=1 -z_dim=2 -beta=1 -KL=True -fadein=True -coff=0.5 -epoch_size=30 -batch_size=128 -checkpoint_dir=checkpoints/dsprite_progress_z2

# training for celebA
python main.py -train_seq=1 -mode=1 -z_dim=7 -beta=1 -KL=True -fadein=True -coff=0.1 -epoch_size=5 -batch_size=128 -checkpoint_dir=checkpoints/Celeb_progress_z7_exp1
python main.py -train_seq=2 -mode=1 -z_dim=7 -beta=1 -KL=True -fadein=True -coff=0.1 -epoch_size=5 -batch_size=128 -checkpoint_dir=checkpoints/Celeb_progress_z7_exp1
python main.py -train_seq=3 -mode=1 -z_dim=7 -beta=1 -KL=True -fadein=True -coff=0.1 -epoch_size=5 -batch_size=128 -checkpoint_dir=checkpoints/Celeb_progress_z7_exp1
python main.py -train_seq=4 -mode=1 -z_dim=7 -beta=1 -KL=True -fadein=True -coff=0.1 -epoch_size=5 -batch_size=128 -checkpoint_dir=checkpoints/Celeb_progress_z7_exp1