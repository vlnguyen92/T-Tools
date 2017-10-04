## Dependencies

- Python 3.5
- TensorFlow 1.1.0 or higher

## Project structure

### Python package `ttools` (Tensorflow-Tools)

Used Tensorflow `tfrecords` to construct pipelines that provide data during training and testing.

Current models:
- LeNet
- ResNet
- SingleLayerCAE
- VGG

Current datasets:
- MNIST
- CIFAR-10

Adding models will require implementing the `_build_model` function. Similarly, a new `Dataset` 
should perform the following: downloading the raw data, preprocessing it, then writing it to tfrecords.

### Scripts

```
usage: python train.py [--dataset {MNIST, Cifar10}]
                       [--model {ResNet, LeNet, SingleLayerCAE, VGG}]
                       [--num_steps TRAIN_STEPS]
                       [--batch_size BATCH_SIZE]
                       [--num_gpus {0, 1}]

optional arguments:
  --dataset {MNIST, Cifar10}
                        The dataset with which the model is trained
  --model {ResNet, Lenet, SingleLayerCAE, VGG}
                        The model name
  --num_steps TRAIN_STEPS
                        Number of training steps
  --batch_size BATCH_SIZE
                        batch size
  --num_gpus {0, 1}
                        0 to train on CPU, 1 to train on GPU
```

### To-Do
- [ ] Add docstrings
- [ ] Add dev set to datasets
- [ ] Add eval script
- [ ] Add tests (create/save models)
- [ ] Install as a package


