# PuzzleGAN: Generating Images Similar to Building Jigsaw Puzzles

Enabling generative adversarial networks (GANs) to learn the distribution of images as compositions of distributions of smaller parts

For more information, please see the paper 
**Face Images as Jigsaw Puzzles: Compositional Perception of Human Faces for Machines Using Generative Adversarial Networks** at
http://arxiv.org/abs/2103.06331

## Preparing datasets for training

The training and evaluation scripts operate on datasets stored as multi-resolution TFRecords. Each dataset is represented by a directory containing the same image data in several resolutions to enable efficient streaming. There is a separate *.tfrecords file for each resolution, and if the dataset contains labels, they are stored in a separate file as well. By default, the scripts expect to find the datasets at `datasets/<NAME>/<NAME>-<RESOLUTION>.tfrecords`. The directory can be changed by editing [config.py](./config.py):

```
result_dir = 'results'
data_dir = 'datasets'
cache_dir = 'cache'
```

To obtain the CelebA-HQ dataset (`datasets/celebahq`), please refer to the [Progressive GAN repository](https://github.com/tkarras/progressive_growing_of_gans).

To obtain other datasets, including LSUN, please consult their corresponding project pages. The datasets can be converted to multi-resolution TFRecords using the provided [dataset_tool.py](./dataset_tool.py):

```
> python dataset_tool.py create_lsun datasets/lsun-bedroom-full ~/lsun/bedroom_lmdb --resolution 128
> python dataset_tool.py create_from_images datasets/custom-dataset ~/custom-images
```

## Training networks

Once the datasets are set up, you can train your own PuzzleGAN network as follows:

1. Edit [train.py](./train.py) to specify one of the puzzle modes and training configuration by uncommenting or editing specific lines.
2. Run the training script with `python train.py`.
3. The results are written to a newly created directory `results/<ID>-<DESCRIPTION>`.

## Puzzle Modes

* Mode 1: Face images composed of two parts; one for the face and one for everything else
<p align="center">
  <img src="https://github.com/MahlaAb/puzzlegan/blob/master/figures/faces_2parts.jpg" width="900">
  <img src="https://github.com/MahlaAb/puzzlegan/blob/master/figures/faces_2parts_swap_example.png" width="600">
</p>



* Mode 2: Face images composed of five parts

<p align="center">
  <img src="https://github.com/MahlaAb/puzzlegan/blob/master/figures/faces_5parts.jpg" width="900">
  <img src="https://github.com/MahlaAb/puzzlegan/blob/master/figures/faces_5parts_swap_examples.png" width="600">
</p>


* Mode 3: Bedroom images composed of two parts
<p align="center">
  <img src="https://github.com/MahlaAb/puzzlegan/blob/master/figures/bedroom_2parts.jpg" width="900">
  <img src="https://github.com/MahlaAb/puzzlegan/blob/master/figures/bedroom_2parts_swap_examples.jpg" width="600">
</p>

* Mode 4: Handwritten digits composed of four parts
<p align="center">
  <img src="https://github.com/MahlaAb/puzzlegan/blob/master/figures/mnist_4parts.jpg" width="600">
  <img src="https://github.com/MahlaAb/puzzlegan/blob/master/figures/mnist_4parts_swap_examples2.jpg" width="600">
</p>

