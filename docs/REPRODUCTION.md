## Train

All the training details (hyper-parameters, trained layers, backbones, etc.) strictly follow the original MatConvNet version of NetVLAD and SARE. 

In order to accomodate computation efficiency, we use small channel dimensions (64*3) and feature sizes of 128/2 X 128/2. We encourage you to change the channel dimensions (`L_DIM`, `M_DIM`, `H_DIM`) and feature sizes to increase the performance.

```shell
bash train.sh
```

The default scripts adopt `4` V100 GPUs (require ~32G per GPU) for training, where each GPU loads one tuple (anchor, positive(s), negatives).
+ In case you want to fasten training, enlarge `GPUS` for more GPUs, or enlarge the `--tuple-size` for more tuples on one GPU;
+ In case your GPU does not have enough memory (e.g. <32G), reduce `--neg-num` for fewer positives or negatives in one tuple.

## Test

During testing, the python scripts will automatically compute the PCA weights from Pitts30k-train or directly load from local files. Generally, `model_best.pth` which is selected by validation in the training performs the best.

Run the test script with `<MODEL PATH>`, e.g. 'logs/saved_models/pitts30k-vgg16/conv5-sare_ind-lr0.001-tuple1/model_best.pth'

```shell
bash test.sh <MODEL PATH>
```

The default scripts adopt `4` V100 GPUs (require ~32G per GPU) for testing.
+ In case you want to fasten training, enlarge `GPUS` for more GPUs, or enlarge the `--test-batch-size` for larger batch size on one GPU;
+ In case your GPU does not have enough memory (e.g. <32G), reduce `--test-batch-size` for smaller batch size on one GPU.