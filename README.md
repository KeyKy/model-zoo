#model-zoo

This repository contains deep learning models. Keep updating....

* Top-1, Top-5 are accuracy in imagenet val which I get from Internet.

* C -> Caffe, M -> MXNet, C2 -> Caffe2


| Model | Top-1 | Top-5 | Caffe | MXNet | Caffe2 | TF |
|:-----|:------:|------:|------:|------:|------:|------:|
| alexnet | C_57.1 | C_80.2 | [243.86M](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) |  |  | |
| caffenet | C_57.4 M_54.5 | C_80.4 M_78.3 | [243.86M](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) | [233M](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-caffenet.md) |  | |
| NIN | C_59.36 M_58.8 | M_81.3 | [29M](https://gist.github.com/mavenlin/d802a5849de39225bcc6) | [29M](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-nin.md) | |
| SqueezeNet1.0| M_55.4 | C_>=80.3 M_78.8 | [4.8M](https://github.com/DeepScale/SqueezeNet) | [4.8M](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-squeezenet.md) | | |
| SqueezeNet1.1 2.4x less computation than 1.0 | C_sameAs1.0 | C_>=80.3 | [4.7M](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1)  | [4.7M](http://data.dmlc.ml/models/imagenet/squeezenet/) | | |
| VGG16 | M_71.0 | C_92.5 M_89.9 | [528M](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) | [528M](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-vgg.md) | | |
| VGG19 | M_71.0 | C_92.5 M_89.8 | [548M](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) | [548M](http://data.dmlc.ml/models/imagenet/vgg/) | | |
| GoogleNet | C_68.7 | C_88.9 | [53.53M](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) | | | |
| Inception-v2(Inception-bn) | M_72.5 | M_90.8 | [134.6M ImageNet21k](https://github.com/pertusa/InceptionBN-21K-for-Caffe)| [43M ImageNet10k](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-bn.md) | | |
| Inception-V3 | M_76.88 | M_93.344 | | [95.6M](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-v3.md)| | |
| ResNet-18 | C_69 M_69.52 | M_89 M_89.08 | [45M](https://github.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet) | [45M](https://github.com/tornadomeet/ResNet) | | |
| ResNet-34 | M_72.8 | M_91.14 | | [83M](https://github.com/tornadomeet/ResNet) | | |
| ResNet-50 | C_75.3 M_75.61 | C_92.2 M_92.76 | [98M](https://github.com/KaimingHe/deep-residual-networks) | [98M](https://github.com/tornadomeet/ResNet) | | |
| ResNet-101 | C_76.4 M_77.32 | C_92.9 M_93.42 | [171M](https://github.com/KaimingHe/deep-residual-networks) | [170M](https://github.com/tornadomeet/ResNet) | | |
| ResNet-152 | C_77.0 M_77.75 | C_93.3 M_93.58 | [231M](https://github.com/KaimingHe/deep-residual-networks) | [230M](https://github.com/tornadomeet/ResNet)| | |
| ResNet-200 | M_77.86 | M_93.84 |  | [247](https://github.com/tornadomeet/ResNet) | | |
|  | | | | | | |

