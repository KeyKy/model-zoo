# TOP1 80.13
# TOP5 95.18
python score.py --prefix /data2/obj_detect/imagenet_models/deploy_inception_resnet_v2 --epoch 0 --gpus 2,3 --batch-size 32 --rgb-mean 128,128,128 --image-shape 3,299,299 --data-val /data_shared/datasets/ILSVRC2015/rec/val_480_q100.rec
