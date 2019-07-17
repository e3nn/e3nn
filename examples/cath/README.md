The scripts in this directory imports this directory as a module.
Therefore you need to set the `PYTHONPATH` to the root project directory.

```
# if pwd is experiments/scripts/cath
PYTHONPATH=../../.. python cath.py --data-filename cath_10arch_ca_reducedx32.npz --model SE3ResNet34Small --training-epochs 1600 --batch-size 16 --batchsize-multiplier 1 --randomize-orientation --kernel-size 3 --initial_lr=0.1 --lr_decay_start=640 --burnin-epochs 640 --lr_decay_base=.996 --downsample-by-pooling --p-drop-conv 0.1 --report-frequency 16 --lamb_conv_weight_L1 1e-7 --lamb_conv_weight_L2 1e-7 --lamb_bn_weight_L1 1e-7 --lamb_bn_weight_L2 1e-7 --report-on-test-set
```