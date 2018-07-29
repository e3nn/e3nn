# SHREC17 Experiment

## training
```bash
python train.py --log_dir myrun --model_path model_deep.py --dataset train --batch_size 32 --learning_rate 0.1 
```
The script will automatically 
- download the dataset (and fix the .obj files)
- load the model given by `model_path` (must be named `Model`)
- train for 2000 epoch with two steps of learning rate decay (each epoch is small because we [equalize the classes](https://github.com/mariogeiger/se3cnn/blob/master/se3cnn/util/dataset/shapes.py#L34-L60))

additional optional arguments are 
- `augmentation` (default: 1) the number of data augmentation by rotation.
- `num_workers` (default: 1) the number of workers for the dataloader. The augmentations are cached therefore we recommend this setting to be set to the number of CPUs until all the cached files are created.
- `restore_dir` (default: None) the path to a previous log directory to resume

## testing
```bash
python test.py --log_dir myrun --dataset val --batch_size 32
```
The script will automatically 
- download the dataset
- load the model saved in `log_dir` and restore its state
- evaluate all the dataset, `dataset` can be either `train`, `val` or `test`
- download and execute [the evaluation script](https://shapenet.cs.stanford.edu/shrec17/)

additional optional arguments are 
- `augmentation` (default: 1) the number of data augmentation by rotation. To avoid **out of memory** keep `augmentation * batch_size` low.
- `num_workers` (default: 1) the number of workers for the dataloader.
