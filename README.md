# proVLAE

Code for paper [PROGRESSIVE LEARNING AND DISENTANGLEMENT OF HIERARCHICAL REPRESENTATIONS](https://openreview.net/forum?id=SJxpsxrYPS).

## Requirements:
python 3.5 \
tensorflow 1.13.1 (also tested on 1.8.0, should be flexible)\
pytorch 0.4.1 (for computing metrics only)\
numpy \
scipy 1.1.0 (You might encounter errors of "imsave" because "imsave is deprecated in SciPy 1.0.0, and will be removed in 1.2.") \
moviepy (for generating .gif images)

## Dataset
dSprite https://github.com/deepmind/dsprites-dataset \
3DShapes https://github.com/deepmind/3d-shapes \
celebA http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (Align&Cropped Images) \
please download those datasets and change the paths in "data_manager.py"

## Scripts
train model: train_pro-VLAE.sh \
view results: visualization.sh \
compute metrics: compute_metric.sh