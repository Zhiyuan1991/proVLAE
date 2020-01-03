# proVLAE

code for paper

## enviroment:
CUDA 10.0.130
python 3.5 \
tensorflow 1.13.1 \
pytorch 0.4.1 \
numpy \
scipy 1.1.0 (You might encounter error of "imsave" because "imsave is deprecated in SciPy 1.0.0, and will be removed in 1.2.")

## dataset
dSprite https://github.com/deepmind/dsprites-dataset \
3DShapes https://github.com/deepmind/3d-shapes \
celebA http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (Align&Cropped Images) \
please downloaded those datasets and change the path in "data_manager.py"
