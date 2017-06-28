# TOP1 79.835
# TOP5 94.695
python score.py --prefix /data2/obj_detect/imagenet_models/inception_v4/deploy_inception_v4 --epoch 0 --gpus 2,3 --batch-size 64 --rgb-mean 128,128,128 --image-shape 3,299,299 --data-val /data_shared/datasets/ILSVRC2015/rec/val_480_q100.rec

