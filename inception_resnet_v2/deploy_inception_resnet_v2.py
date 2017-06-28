import mxnet as mx
data = mx.symbol.Variable(name='data')
conv1_3x3_s2 = mx.symbol.Convolution(name='conv1_3x3_s2', data=data , num_filter=32, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
conv1_3x3_s2_bn = mx.symbol.BatchNorm(name='conv1_3x3_s2_bn', data=conv1_3x3_s2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv1_3x3_s2_scale = conv1_3x3_s2_bn
conv1_3x3_relu = mx.symbol.Activation(name='conv1_3x3_relu', data=conv1_3x3_s2_scale , act_type='relu')
conv2_3x3_s1 = mx.symbol.Convolution(name='conv2_3x3_s1', data=conv1_3x3_relu , num_filter=32, pad=(0, 0), kernel=(3,3), stride=(1,1), no_bias=True)
conv2_3x3_s1_bn = mx.symbol.BatchNorm(name='conv2_3x3_s1_bn', data=conv2_3x3_s1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv2_3x3_s1_scale = conv2_3x3_s1_bn
conv2_3x3_relu = mx.symbol.Activation(name='conv2_3x3_relu', data=conv2_3x3_s1_scale , act_type='relu')
conv3_3x3_s1 = mx.symbol.Convolution(name='conv3_3x3_s1', data=conv2_3x3_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv3_3x3_s1_bn = mx.symbol.BatchNorm(name='conv3_3x3_s1_bn', data=conv3_3x3_s1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv3_3x3_s1_scale = conv3_3x3_s1_bn
conv3_3x3_relu = mx.symbol.Activation(name='conv3_3x3_relu', data=conv3_3x3_s1_scale , act_type='relu')
pool1_3x3_s2 = mx.symbol.Pooling(name='pool1_3x3_s2', data=conv3_3x3_relu , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
conv4_3x3_reduce = mx.symbol.Convolution(name='conv4_3x3_reduce', data=pool1_3x3_s2 , num_filter=80, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv4_3x3_reduce_bn = mx.symbol.BatchNorm(name='conv4_3x3_reduce_bn', data=conv4_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv4_3x3_reduce_scale = conv4_3x3_reduce_bn
conv4_3x3_reduce_relu = mx.symbol.Activation(name='conv4_3x3_reduce_relu', data=conv4_3x3_reduce_scale , act_type='relu')
conv4_3x3 = mx.symbol.Convolution(name='conv4_3x3', data=conv4_3x3_reduce_relu , num_filter=192, pad=(0, 0), kernel=(3,3), stride=(1,1), no_bias=True)
conv4_3x3_bn = mx.symbol.BatchNorm(name='conv4_3x3_bn', data=conv4_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv4_3x3_scale = conv4_3x3_bn
conv4_relu_3x3 = mx.symbol.Activation(name='conv4_relu_3x3', data=conv4_3x3_scale , act_type='relu')
pool2_3x3_s2 = mx.symbol.Pooling(name='pool2_3x3_s2', data=conv4_relu_3x3 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
conv5_1x1 = mx.symbol.Convolution(name='conv5_1x1', data=pool2_3x3_s2 , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv5_1x1_bn = mx.symbol.BatchNorm(name='conv5_1x1_bn', data=conv5_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv5_1x1_scale = conv5_1x1_bn
conv5_1x1_relu = mx.symbol.Activation(name='conv5_1x1_relu', data=conv5_1x1_scale , act_type='relu')
conv5_5x5_reduce = mx.symbol.Convolution(name='conv5_5x5_reduce', data=pool2_3x3_s2 , num_filter=48, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv5_5x5_reduce_bn = mx.symbol.BatchNorm(name='conv5_5x5_reduce_bn', data=conv5_5x5_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv5_5x5_reduce_scale = conv5_5x5_reduce_bn
conv5_5x5_reduce_relu = mx.symbol.Activation(name='conv5_5x5_reduce_relu', data=conv5_5x5_reduce_scale , act_type='relu')
conv5_5x5 = mx.symbol.Convolution(name='conv5_5x5', data=conv5_5x5_reduce_relu , num_filter=64, pad=(2, 2), kernel=(5,5), stride=(1,1), no_bias=True)
conv5_5x5_bn = mx.symbol.BatchNorm(name='conv5_5x5_bn', data=conv5_5x5 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv5_5x5_scale = conv5_5x5_bn
conv5_5x5_relu = mx.symbol.Activation(name='conv5_5x5_relu', data=conv5_5x5_scale , act_type='relu')
conv5_3x3_reduce = mx.symbol.Convolution(name='conv5_3x3_reduce', data=pool2_3x3_s2 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv5_3x3_reduce_bn = mx.symbol.BatchNorm(name='conv5_3x3_reduce_bn', data=conv5_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv5_3x3_reduce_scale = conv5_3x3_reduce_bn
conv5_3x3_reduce_relu = mx.symbol.Activation(name='conv5_3x3_reduce_relu', data=conv5_3x3_reduce_scale , act_type='relu')
conv5_3x3 = mx.symbol.Convolution(name='conv5_3x3', data=conv5_3x3_reduce_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv5_3x3_bn = mx.symbol.BatchNorm(name='conv5_3x3_bn', data=conv5_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv5_3x3_scale = conv5_3x3_bn
conv5_3x3_relu = mx.symbol.Activation(name='conv5_3x3_relu', data=conv5_3x3_scale , act_type='relu')
conv5_3x3_2 = mx.symbol.Convolution(name='conv5_3x3_2', data=conv5_3x3_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv5_3x3_2_bn = mx.symbol.BatchNorm(name='conv5_3x3_2_bn', data=conv5_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv5_3x3_2_scale = conv5_3x3_2_bn
conv5_3x3_2_relu = mx.symbol.Activation(name='conv5_3x3_2_relu', data=conv5_3x3_2_scale , act_type='relu')
ave_pool = mx.symbol.Pooling(name='ave_pool', data=pool2_3x3_s2 , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
conv5_1x1_ave = mx.symbol.Convolution(name='conv5_1x1_ave', data=ave_pool , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv5_1x1_ave_bn = mx.symbol.BatchNorm(name='conv5_1x1_ave_bn', data=conv5_1x1_ave , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv5_1x1_ave_scale = conv5_1x1_ave_bn
conv5_1x1_ave_relu = mx.symbol.Activation(name='conv5_1x1_ave_relu', data=conv5_1x1_ave_scale , act_type='relu')
stem_concat = mx.symbol.Concat(name='stem_concat', *[conv5_1x1_relu,conv5_5x5_relu,conv5_3x3_2_relu,conv5_1x1_ave_relu] )
inception_resnet_v2_a1_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a1_1x1', data=stem_concat , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a1_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a1_1x1_bn', data=inception_resnet_v2_a1_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a1_1x1_scale = inception_resnet_v2_a1_1x1_bn
inception_resnet_v2_a1_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a1_1x1_relu', data=inception_resnet_v2_a1_1x1_scale , act_type='relu')
inception_resnet_v2_a1_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a1_3x3_reduce', data=stem_concat , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a1_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a1_3x3_reduce_bn', data=inception_resnet_v2_a1_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a1_3x3_reduce_scale = inception_resnet_v2_a1_3x3_reduce_bn
inception_resnet_v2_a1_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a1_3x3_reduce_relu', data=inception_resnet_v2_a1_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a1_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a1_3x3', data=inception_resnet_v2_a1_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a1_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a1_3x3_bn', data=inception_resnet_v2_a1_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a1_3x3_scale = inception_resnet_v2_a1_3x3_bn
inception_resnet_v2_a1_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a1_3x3_relu', data=inception_resnet_v2_a1_3x3_scale , act_type='relu')
inception_resnet_v2_a1_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a1_3x3_2_reduce', data=stem_concat , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a1_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a1_3x3_2_reduce_bn', data=inception_resnet_v2_a1_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a1_3x3_2_reduce_scale = inception_resnet_v2_a1_3x3_2_reduce_bn
inception_resnet_v2_a1_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a1_3x3_2_reduce_relu', data=inception_resnet_v2_a1_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a1_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a1_3x3_2', data=inception_resnet_v2_a1_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a1_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a1_3x3_2_bn', data=inception_resnet_v2_a1_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a1_3x3_2_scale = inception_resnet_v2_a1_3x3_2_bn
inception_resnet_v2_a1_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a1_3x3_2_relu', data=inception_resnet_v2_a1_3x3_2_scale , act_type='relu')
inception_resnet_v2_a1_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a1_3x3_3', data=inception_resnet_v2_a1_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a1_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a1_3x3_3_bn', data=inception_resnet_v2_a1_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a1_3x3_3_scale = inception_resnet_v2_a1_3x3_3_bn
inception_resnet_v2_a1_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a1_3x3_3_relu', data=inception_resnet_v2_a1_3x3_3_scale , act_type='relu')
inception_resnet_v2_a1_concat = mx.symbol.Concat(name='inception_resnet_v2_a1_concat', *[inception_resnet_v2_a1_1x1_relu,inception_resnet_v2_a1_3x3_relu,inception_resnet_v2_a1_3x3_3_relu] )
inception_resnet_v2_a1_up = mx.symbol.Convolution(name='inception_resnet_v2_a1_up', data=inception_resnet_v2_a1_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a1_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a1_residual_eltwise', *[1.0*stem_concat,0.170000001788*inception_resnet_v2_a1_up] )
inception_resnet_v2_a1_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a1_residual_eltwise_relu', data=inception_resnet_v2_a1_residual_eltwise , act_type='relu')
inception_resnet_v2_a2_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a2_1x1', data=inception_resnet_v2_a1_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a2_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a2_1x1_bn', data=inception_resnet_v2_a2_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a2_1x1_scale = inception_resnet_v2_a2_1x1_bn
inception_resnet_v2_a2_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a2_1x1_relu', data=inception_resnet_v2_a2_1x1_scale , act_type='relu')
inception_resnet_v2_a2_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a2_3x3_reduce', data=inception_resnet_v2_a1_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a2_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a2_3x3_reduce_bn', data=inception_resnet_v2_a2_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a2_3x3_reduce_scale = inception_resnet_v2_a2_3x3_reduce_bn
inception_resnet_v2_a2_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a2_3x3_reduce_relu', data=inception_resnet_v2_a2_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a2_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a2_3x3', data=inception_resnet_v2_a2_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a2_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a2_3x3_bn', data=inception_resnet_v2_a2_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a2_3x3_scale = inception_resnet_v2_a2_3x3_bn
inception_resnet_v2_a2_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a2_3x3_relu', data=inception_resnet_v2_a2_3x3_scale , act_type='relu')
inception_resnet_v2_a2_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a2_3x3_2_reduce', data=inception_resnet_v2_a1_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a2_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a2_3x3_2_reduce_bn', data=inception_resnet_v2_a2_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a2_3x3_2_reduce_scale = inception_resnet_v2_a2_3x3_2_reduce_bn
inception_resnet_v2_a2_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a2_3x3_2_reduce_relu', data=inception_resnet_v2_a2_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a2_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a2_3x3_2', data=inception_resnet_v2_a2_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a2_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a2_3x3_2_bn', data=inception_resnet_v2_a2_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a2_3x3_2_scale = inception_resnet_v2_a2_3x3_2_bn
inception_resnet_v2_a2_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a2_3x3_2_relu', data=inception_resnet_v2_a2_3x3_2_scale , act_type='relu')
inception_resnet_v2_a2_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a2_3x3_3', data=inception_resnet_v2_a2_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a2_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a2_3x3_3_bn', data=inception_resnet_v2_a2_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a2_3x3_3_scale = inception_resnet_v2_a2_3x3_3_bn
inception_resnet_v2_a2_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a2_3x3_3_relu', data=inception_resnet_v2_a2_3x3_3_scale , act_type='relu')
inception_resnet_v2_a2_concat = mx.symbol.Concat(name='inception_resnet_v2_a2_concat', *[inception_resnet_v2_a2_1x1_relu,inception_resnet_v2_a2_3x3_relu,inception_resnet_v2_a2_3x3_3_relu] )
inception_resnet_v2_a2_up = mx.symbol.Convolution(name='inception_resnet_v2_a2_up', data=inception_resnet_v2_a2_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a2_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a2_residual_eltwise', *[1.0*inception_resnet_v2_a1_residual_eltwise_relu,0.170000001788*inception_resnet_v2_a2_up] )
inception_resnet_v2_a2_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a2_residual_eltwise_relu', data=inception_resnet_v2_a2_residual_eltwise , act_type='relu')
inception_resnet_v2_a3_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a3_1x1', data=inception_resnet_v2_a2_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a3_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a3_1x1_bn', data=inception_resnet_v2_a3_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a3_1x1_scale = inception_resnet_v2_a3_1x1_bn
inception_resnet_v2_a3_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a3_1x1_relu', data=inception_resnet_v2_a3_1x1_scale , act_type='relu')
inception_resnet_v2_a3_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a3_3x3_reduce', data=inception_resnet_v2_a2_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a3_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a3_3x3_reduce_bn', data=inception_resnet_v2_a3_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a3_3x3_reduce_scale = inception_resnet_v2_a3_3x3_reduce_bn
inception_resnet_v2_a3_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a3_3x3_reduce_relu', data=inception_resnet_v2_a3_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a3_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a3_3x3', data=inception_resnet_v2_a3_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a3_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a3_3x3_bn', data=inception_resnet_v2_a3_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a3_3x3_scale = inception_resnet_v2_a3_3x3_bn
inception_resnet_v2_a3_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a3_3x3_relu', data=inception_resnet_v2_a3_3x3_scale , act_type='relu')
inception_resnet_v2_a3_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a3_3x3_2_reduce', data=inception_resnet_v2_a2_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a3_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a3_3x3_2_reduce_bn', data=inception_resnet_v2_a3_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a3_3x3_2_reduce_scale = inception_resnet_v2_a3_3x3_2_reduce_bn
inception_resnet_v2_a3_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a3_3x3_2_reduce_relu', data=inception_resnet_v2_a3_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a3_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a3_3x3_2', data=inception_resnet_v2_a3_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a3_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a3_3x3_2_bn', data=inception_resnet_v2_a3_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a3_3x3_2_scale = inception_resnet_v2_a3_3x3_2_bn
inception_resnet_v2_a3_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a3_3x3_2_relu', data=inception_resnet_v2_a3_3x3_2_scale , act_type='relu')
inception_resnet_v2_a3_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a3_3x3_3', data=inception_resnet_v2_a3_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a3_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a3_3x3_3_bn', data=inception_resnet_v2_a3_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a3_3x3_3_scale = inception_resnet_v2_a3_3x3_3_bn
inception_resnet_v2_a3_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a3_3x3_3_relu', data=inception_resnet_v2_a3_3x3_3_scale , act_type='relu')
inception_resnet_v2_a3_concat = mx.symbol.Concat(name='inception_resnet_v2_a3_concat', *[inception_resnet_v2_a3_1x1_relu,inception_resnet_v2_a3_3x3_relu,inception_resnet_v2_a3_3x3_3_relu] )
inception_resnet_v2_a3_up = mx.symbol.Convolution(name='inception_resnet_v2_a3_up', data=inception_resnet_v2_a3_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a3_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a3_residual_eltwise', *[1.0*inception_resnet_v2_a2_residual_eltwise_relu,0.170000001788*inception_resnet_v2_a3_up] )
inception_resnet_v2_a3_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a3_residual_eltwise_relu', data=inception_resnet_v2_a3_residual_eltwise , act_type='relu')
inception_resnet_v2_a4_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a4_1x1', data=inception_resnet_v2_a3_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a4_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a4_1x1_bn', data=inception_resnet_v2_a4_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a4_1x1_scale = inception_resnet_v2_a4_1x1_bn
inception_resnet_v2_a4_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a4_1x1_relu', data=inception_resnet_v2_a4_1x1_scale , act_type='relu')
inception_resnet_v2_a4_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a4_3x3_reduce', data=inception_resnet_v2_a3_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a4_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a4_3x3_reduce_bn', data=inception_resnet_v2_a4_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a4_3x3_reduce_scale = inception_resnet_v2_a4_3x3_reduce_bn
inception_resnet_v2_a4_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a4_3x3_reduce_relu', data=inception_resnet_v2_a4_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a4_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a4_3x3', data=inception_resnet_v2_a4_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a4_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a4_3x3_bn', data=inception_resnet_v2_a4_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a4_3x3_scale = inception_resnet_v2_a4_3x3_bn
inception_resnet_v2_a4_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a4_3x3_relu', data=inception_resnet_v2_a4_3x3_scale , act_type='relu')
inception_resnet_v2_a4_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a4_3x3_2_reduce', data=inception_resnet_v2_a3_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a4_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a4_3x3_2_reduce_bn', data=inception_resnet_v2_a4_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a4_3x3_2_reduce_scale = inception_resnet_v2_a4_3x3_2_reduce_bn
inception_resnet_v2_a4_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a4_3x3_2_reduce_relu', data=inception_resnet_v2_a4_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a4_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a4_3x3_2', data=inception_resnet_v2_a4_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a4_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a4_3x3_2_bn', data=inception_resnet_v2_a4_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a4_3x3_2_scale = inception_resnet_v2_a4_3x3_2_bn
inception_resnet_v2_a4_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a4_3x3_2_relu', data=inception_resnet_v2_a4_3x3_2_scale , act_type='relu')
inception_resnet_v2_a4_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a4_3x3_3', data=inception_resnet_v2_a4_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a4_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a4_3x3_3_bn', data=inception_resnet_v2_a4_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a4_3x3_3_scale = inception_resnet_v2_a4_3x3_3_bn
inception_resnet_v2_a4_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a4_3x3_3_relu', data=inception_resnet_v2_a4_3x3_3_scale , act_type='relu')
inception_resnet_v2_a4_concat = mx.symbol.Concat(name='inception_resnet_v2_a4_concat', *[inception_resnet_v2_a4_1x1_relu,inception_resnet_v2_a4_3x3_relu,inception_resnet_v2_a4_3x3_3_relu] )
inception_resnet_v2_a4_up = mx.symbol.Convolution(name='inception_resnet_v2_a4_up', data=inception_resnet_v2_a4_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a4_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a4_residual_eltwise', *[1.0*inception_resnet_v2_a3_residual_eltwise_relu,0.170000001788*inception_resnet_v2_a4_up] )
inception_resnet_v2_a4_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a4_residual_eltwise_relu', data=inception_resnet_v2_a4_residual_eltwise , act_type='relu')
inception_resnet_v2_a5_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a5_1x1', data=inception_resnet_v2_a4_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a5_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a5_1x1_bn', data=inception_resnet_v2_a5_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a5_1x1_scale = inception_resnet_v2_a5_1x1_bn
inception_resnet_v2_a5_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a5_1x1_relu', data=inception_resnet_v2_a5_1x1_scale , act_type='relu')
inception_resnet_v2_a5_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a5_3x3_reduce', data=inception_resnet_v2_a4_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a5_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a5_3x3_reduce_bn', data=inception_resnet_v2_a5_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a5_3x3_reduce_scale = inception_resnet_v2_a5_3x3_reduce_bn
inception_resnet_v2_a5_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a5_3x3_reduce_relu', data=inception_resnet_v2_a5_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a5_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a5_3x3', data=inception_resnet_v2_a5_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a5_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a5_3x3_bn', data=inception_resnet_v2_a5_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a5_3x3_scale = inception_resnet_v2_a5_3x3_bn
inception_resnet_v2_a5_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a5_3x3_relu', data=inception_resnet_v2_a5_3x3_scale , act_type='relu')
inception_resnet_v2_a5_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a5_3x3_2_reduce', data=inception_resnet_v2_a4_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a5_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a5_3x3_2_reduce_bn', data=inception_resnet_v2_a5_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a5_3x3_2_reduce_scale = inception_resnet_v2_a5_3x3_2_reduce_bn
inception_resnet_v2_a5_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a5_3x3_2_reduce_relu', data=inception_resnet_v2_a5_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a5_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a5_3x3_2', data=inception_resnet_v2_a5_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a5_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a5_3x3_2_bn', data=inception_resnet_v2_a5_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a5_3x3_2_scale = inception_resnet_v2_a5_3x3_2_bn
inception_resnet_v2_a5_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a5_3x3_2_relu', data=inception_resnet_v2_a5_3x3_2_scale , act_type='relu')
inception_resnet_v2_a5_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a5_3x3_3', data=inception_resnet_v2_a5_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a5_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a5_3x3_3_bn', data=inception_resnet_v2_a5_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a5_3x3_3_scale = inception_resnet_v2_a5_3x3_3_bn
inception_resnet_v2_a5_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a5_3x3_3_relu', data=inception_resnet_v2_a5_3x3_3_scale , act_type='relu')
inception_resnet_v2_a5_concat = mx.symbol.Concat(name='inception_resnet_v2_a5_concat', *[inception_resnet_v2_a5_1x1_relu,inception_resnet_v2_a5_3x3_relu,inception_resnet_v2_a5_3x3_3_relu] )
inception_resnet_v2_a5_up = mx.symbol.Convolution(name='inception_resnet_v2_a5_up', data=inception_resnet_v2_a5_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a5_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a5_residual_eltwise', *[1.0*inception_resnet_v2_a4_residual_eltwise_relu,0.170000001788*inception_resnet_v2_a5_up] )
inception_resnet_v2_a5_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a5_residual_eltwise_relu', data=inception_resnet_v2_a5_residual_eltwise , act_type='relu')
inception_resnet_v2_a6_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a6_1x1', data=inception_resnet_v2_a5_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a6_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a6_1x1_bn', data=inception_resnet_v2_a6_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a6_1x1_scale = inception_resnet_v2_a6_1x1_bn
inception_resnet_v2_a6_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a6_1x1_relu', data=inception_resnet_v2_a6_1x1_scale , act_type='relu')
inception_resnet_v2_a6_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a6_3x3_reduce', data=inception_resnet_v2_a5_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a6_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a6_3x3_reduce_bn', data=inception_resnet_v2_a6_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a6_3x3_reduce_scale = inception_resnet_v2_a6_3x3_reduce_bn
inception_resnet_v2_a6_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a6_3x3_reduce_relu', data=inception_resnet_v2_a6_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a6_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a6_3x3', data=inception_resnet_v2_a6_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a6_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a6_3x3_bn', data=inception_resnet_v2_a6_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a6_3x3_scale = inception_resnet_v2_a6_3x3_bn
inception_resnet_v2_a6_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a6_3x3_relu', data=inception_resnet_v2_a6_3x3_scale , act_type='relu')
inception_resnet_v2_a6_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a6_3x3_2_reduce', data=inception_resnet_v2_a5_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a6_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a6_3x3_2_reduce_bn', data=inception_resnet_v2_a6_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a6_3x3_2_reduce_scale = inception_resnet_v2_a6_3x3_2_reduce_bn
inception_resnet_v2_a6_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a6_3x3_2_reduce_relu', data=inception_resnet_v2_a6_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a6_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a6_3x3_2', data=inception_resnet_v2_a6_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a6_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a6_3x3_2_bn', data=inception_resnet_v2_a6_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a6_3x3_2_scale = inception_resnet_v2_a6_3x3_2_bn
inception_resnet_v2_a6_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a6_3x3_2_relu', data=inception_resnet_v2_a6_3x3_2_scale , act_type='relu')
inception_resnet_v2_a6_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a6_3x3_3', data=inception_resnet_v2_a6_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a6_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a6_3x3_3_bn', data=inception_resnet_v2_a6_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a6_3x3_3_scale = inception_resnet_v2_a6_3x3_3_bn
inception_resnet_v2_a6_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a6_3x3_3_relu', data=inception_resnet_v2_a6_3x3_3_scale , act_type='relu')
inception_resnet_v2_a6_concat = mx.symbol.Concat(name='inception_resnet_v2_a6_concat', *[inception_resnet_v2_a6_1x1_relu,inception_resnet_v2_a6_3x3_relu,inception_resnet_v2_a6_3x3_3_relu] )
inception_resnet_v2_a6_up = mx.symbol.Convolution(name='inception_resnet_v2_a6_up', data=inception_resnet_v2_a6_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a6_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a6_residual_eltwise', *[1.0*inception_resnet_v2_a5_residual_eltwise_relu,0.170000001788*inception_resnet_v2_a6_up] )
inception_resnet_v2_a6_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a6_residual_eltwise_relu', data=inception_resnet_v2_a6_residual_eltwise , act_type='relu')
inception_resnet_v2_a7_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a7_1x1', data=inception_resnet_v2_a6_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a7_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a7_1x1_bn', data=inception_resnet_v2_a7_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a7_1x1_scale = inception_resnet_v2_a7_1x1_bn
inception_resnet_v2_a7_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a7_1x1_relu', data=inception_resnet_v2_a7_1x1_scale , act_type='relu')
inception_resnet_v2_a7_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a7_3x3_reduce', data=inception_resnet_v2_a6_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a7_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a7_3x3_reduce_bn', data=inception_resnet_v2_a7_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a7_3x3_reduce_scale = inception_resnet_v2_a7_3x3_reduce_bn
inception_resnet_v2_a7_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a7_3x3_reduce_relu', data=inception_resnet_v2_a7_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a7_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a7_3x3', data=inception_resnet_v2_a7_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a7_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a7_3x3_bn', data=inception_resnet_v2_a7_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a7_3x3_scale = inception_resnet_v2_a7_3x3_bn
inception_resnet_v2_a7_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a7_3x3_relu', data=inception_resnet_v2_a7_3x3_scale , act_type='relu')
inception_resnet_v2_a7_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a7_3x3_2_reduce', data=inception_resnet_v2_a6_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a7_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a7_3x3_2_reduce_bn', data=inception_resnet_v2_a7_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a7_3x3_2_reduce_scale = inception_resnet_v2_a7_3x3_2_reduce_bn
inception_resnet_v2_a7_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a7_3x3_2_reduce_relu', data=inception_resnet_v2_a7_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a7_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a7_3x3_2', data=inception_resnet_v2_a7_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a7_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a7_3x3_2_bn', data=inception_resnet_v2_a7_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a7_3x3_2_scale = inception_resnet_v2_a7_3x3_2_bn
inception_resnet_v2_a7_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a7_3x3_2_relu', data=inception_resnet_v2_a7_3x3_2_scale , act_type='relu')
inception_resnet_v2_a7_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a7_3x3_3', data=inception_resnet_v2_a7_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a7_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a7_3x3_3_bn', data=inception_resnet_v2_a7_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a7_3x3_3_scale = inception_resnet_v2_a7_3x3_3_bn
inception_resnet_v2_a7_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a7_3x3_3_relu', data=inception_resnet_v2_a7_3x3_3_scale , act_type='relu')
inception_resnet_v2_a7_concat = mx.symbol.Concat(name='inception_resnet_v2_a7_concat', *[inception_resnet_v2_a7_1x1_relu,inception_resnet_v2_a7_3x3_relu,inception_resnet_v2_a7_3x3_3_relu] )
inception_resnet_v2_a7_up = mx.symbol.Convolution(name='inception_resnet_v2_a7_up', data=inception_resnet_v2_a7_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a7_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a7_residual_eltwise', *[1.0*inception_resnet_v2_a6_residual_eltwise_relu,0.170000001788*inception_resnet_v2_a7_up] )
inception_resnet_v2_a7_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a7_residual_eltwise_relu', data=inception_resnet_v2_a7_residual_eltwise , act_type='relu')
inception_resnet_v2_a8_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a8_1x1', data=inception_resnet_v2_a7_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a8_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a8_1x1_bn', data=inception_resnet_v2_a8_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a8_1x1_scale = inception_resnet_v2_a8_1x1_bn
inception_resnet_v2_a8_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a8_1x1_relu', data=inception_resnet_v2_a8_1x1_scale , act_type='relu')
inception_resnet_v2_a8_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a8_3x3_reduce', data=inception_resnet_v2_a7_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a8_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a8_3x3_reduce_bn', data=inception_resnet_v2_a8_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a8_3x3_reduce_scale = inception_resnet_v2_a8_3x3_reduce_bn
inception_resnet_v2_a8_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a8_3x3_reduce_relu', data=inception_resnet_v2_a8_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a8_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a8_3x3', data=inception_resnet_v2_a8_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a8_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a8_3x3_bn', data=inception_resnet_v2_a8_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a8_3x3_scale = inception_resnet_v2_a8_3x3_bn
inception_resnet_v2_a8_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a8_3x3_relu', data=inception_resnet_v2_a8_3x3_scale , act_type='relu')
inception_resnet_v2_a8_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a8_3x3_2_reduce', data=inception_resnet_v2_a7_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a8_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a8_3x3_2_reduce_bn', data=inception_resnet_v2_a8_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a8_3x3_2_reduce_scale = inception_resnet_v2_a8_3x3_2_reduce_bn
inception_resnet_v2_a8_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a8_3x3_2_reduce_relu', data=inception_resnet_v2_a8_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a8_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a8_3x3_2', data=inception_resnet_v2_a8_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a8_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a8_3x3_2_bn', data=inception_resnet_v2_a8_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a8_3x3_2_scale = inception_resnet_v2_a8_3x3_2_bn
inception_resnet_v2_a8_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a8_3x3_2_relu', data=inception_resnet_v2_a8_3x3_2_scale , act_type='relu')
inception_resnet_v2_a8_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a8_3x3_3', data=inception_resnet_v2_a8_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a8_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a8_3x3_3_bn', data=inception_resnet_v2_a8_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a8_3x3_3_scale = inception_resnet_v2_a8_3x3_3_bn
inception_resnet_v2_a8_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a8_3x3_3_relu', data=inception_resnet_v2_a8_3x3_3_scale , act_type='relu')
inception_resnet_v2_a8_concat = mx.symbol.Concat(name='inception_resnet_v2_a8_concat', *[inception_resnet_v2_a8_1x1_relu,inception_resnet_v2_a8_3x3_relu,inception_resnet_v2_a8_3x3_3_relu] )
inception_resnet_v2_a8_up = mx.symbol.Convolution(name='inception_resnet_v2_a8_up', data=inception_resnet_v2_a8_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a8_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a8_residual_eltwise', *[1.0*inception_resnet_v2_a7_residual_eltwise_relu,0.170000001788*inception_resnet_v2_a8_up] )
inception_resnet_v2_a8_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a8_residual_eltwise_relu', data=inception_resnet_v2_a8_residual_eltwise , act_type='relu')
inception_resnet_v2_a9_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a9_1x1', data=inception_resnet_v2_a8_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a9_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a9_1x1_bn', data=inception_resnet_v2_a9_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a9_1x1_scale = inception_resnet_v2_a9_1x1_bn
inception_resnet_v2_a9_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a9_1x1_relu', data=inception_resnet_v2_a9_1x1_scale , act_type='relu')
inception_resnet_v2_a9_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a9_3x3_reduce', data=inception_resnet_v2_a8_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a9_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a9_3x3_reduce_bn', data=inception_resnet_v2_a9_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a9_3x3_reduce_scale = inception_resnet_v2_a9_3x3_reduce_bn
inception_resnet_v2_a9_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a9_3x3_reduce_relu', data=inception_resnet_v2_a9_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a9_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a9_3x3', data=inception_resnet_v2_a9_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a9_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a9_3x3_bn', data=inception_resnet_v2_a9_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a9_3x3_scale = inception_resnet_v2_a9_3x3_bn
inception_resnet_v2_a9_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a9_3x3_relu', data=inception_resnet_v2_a9_3x3_scale , act_type='relu')
inception_resnet_v2_a9_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a9_3x3_2_reduce', data=inception_resnet_v2_a8_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a9_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a9_3x3_2_reduce_bn', data=inception_resnet_v2_a9_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a9_3x3_2_reduce_scale = inception_resnet_v2_a9_3x3_2_reduce_bn
inception_resnet_v2_a9_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a9_3x3_2_reduce_relu', data=inception_resnet_v2_a9_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a9_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a9_3x3_2', data=inception_resnet_v2_a9_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a9_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a9_3x3_2_bn', data=inception_resnet_v2_a9_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a9_3x3_2_scale = inception_resnet_v2_a9_3x3_2_bn
inception_resnet_v2_a9_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a9_3x3_2_relu', data=inception_resnet_v2_a9_3x3_2_scale , act_type='relu')
inception_resnet_v2_a9_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a9_3x3_3', data=inception_resnet_v2_a9_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a9_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a9_3x3_3_bn', data=inception_resnet_v2_a9_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a9_3x3_3_scale = inception_resnet_v2_a9_3x3_3_bn
inception_resnet_v2_a9_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a9_3x3_3_relu', data=inception_resnet_v2_a9_3x3_3_scale , act_type='relu')
inception_resnet_v2_a9_concat = mx.symbol.Concat(name='inception_resnet_v2_a9_concat', *[inception_resnet_v2_a9_1x1_relu,inception_resnet_v2_a9_3x3_relu,inception_resnet_v2_a9_3x3_3_relu] )
inception_resnet_v2_a9_up = mx.symbol.Convolution(name='inception_resnet_v2_a9_up', data=inception_resnet_v2_a9_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a9_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a9_residual_eltwise', *[1.0*inception_resnet_v2_a8_residual_eltwise_relu,0.170000001788*inception_resnet_v2_a9_up] )
inception_resnet_v2_a9_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a9_residual_eltwise_relu', data=inception_resnet_v2_a9_residual_eltwise , act_type='relu')
inception_resnet_v2_a10_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_a10_1x1', data=inception_resnet_v2_a9_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a10_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a10_1x1_bn', data=inception_resnet_v2_a10_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a10_1x1_scale = inception_resnet_v2_a10_1x1_bn
inception_resnet_v2_a10_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_a10_1x1_relu', data=inception_resnet_v2_a10_1x1_scale , act_type='relu')
inception_resnet_v2_a10_3x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a10_3x3_reduce', data=inception_resnet_v2_a9_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a10_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a10_3x3_reduce_bn', data=inception_resnet_v2_a10_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a10_3x3_reduce_scale = inception_resnet_v2_a10_3x3_reduce_bn
inception_resnet_v2_a10_3x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a10_3x3_reduce_relu', data=inception_resnet_v2_a10_3x3_reduce_scale , act_type='relu')
inception_resnet_v2_a10_3x3 = mx.symbol.Convolution(name='inception_resnet_v2_a10_3x3', data=inception_resnet_v2_a10_3x3_reduce_relu , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a10_3x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a10_3x3_bn', data=inception_resnet_v2_a10_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a10_3x3_scale = inception_resnet_v2_a10_3x3_bn
inception_resnet_v2_a10_3x3_relu = mx.symbol.Activation(name='inception_resnet_v2_a10_3x3_relu', data=inception_resnet_v2_a10_3x3_scale , act_type='relu')
inception_resnet_v2_a10_3x3_2_reduce = mx.symbol.Convolution(name='inception_resnet_v2_a10_3x3_2_reduce', data=inception_resnet_v2_a9_residual_eltwise_relu , num_filter=32, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_a10_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a10_3x3_2_reduce_bn', data=inception_resnet_v2_a10_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a10_3x3_2_reduce_scale = inception_resnet_v2_a10_3x3_2_reduce_bn
inception_resnet_v2_a10_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_a10_3x3_2_reduce_relu', data=inception_resnet_v2_a10_3x3_2_reduce_scale , act_type='relu')
inception_resnet_v2_a10_3x3_2 = mx.symbol.Convolution(name='inception_resnet_v2_a10_3x3_2', data=inception_resnet_v2_a10_3x3_2_reduce_relu , num_filter=48, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a10_3x3_2_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a10_3x3_2_bn', data=inception_resnet_v2_a10_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a10_3x3_2_scale = inception_resnet_v2_a10_3x3_2_bn
inception_resnet_v2_a10_3x3_2_relu = mx.symbol.Activation(name='inception_resnet_v2_a10_3x3_2_relu', data=inception_resnet_v2_a10_3x3_2_scale , act_type='relu')
inception_resnet_v2_a10_3x3_3 = mx.symbol.Convolution(name='inception_resnet_v2_a10_3x3_3', data=inception_resnet_v2_a10_3x3_2_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_resnet_v2_a10_3x3_3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_a10_3x3_3_bn', data=inception_resnet_v2_a10_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_a10_3x3_3_scale = inception_resnet_v2_a10_3x3_3_bn
inception_resnet_v2_a10_3x3_3_relu = mx.symbol.Activation(name='inception_resnet_v2_a10_3x3_3_relu', data=inception_resnet_v2_a10_3x3_3_scale , act_type='relu')
inception_resnet_v2_a10_concat = mx.symbol.Concat(name='inception_resnet_v2_a10_concat', *[inception_resnet_v2_a10_1x1_relu,inception_resnet_v2_a10_3x3_relu,inception_resnet_v2_a10_3x3_3_relu] )
inception_resnet_v2_a10_up = mx.symbol.Convolution(name='inception_resnet_v2_a10_up', data=inception_resnet_v2_a10_concat , num_filter=320, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_a10_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_a10_residual_eltwise', *[1.0*inception_resnet_v2_a9_residual_eltwise_relu,0.170000001788*inception_resnet_v2_a10_up] )
inception_resnet_v2_a10_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_a10_residual_eltwise_relu', data=inception_resnet_v2_a10_residual_eltwise , act_type='relu')
reduction_a_3x3 = mx.symbol.Convolution(name='reduction_a_3x3', data=inception_resnet_v2_a10_residual_eltwise_relu , num_filter=384, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
reduction_a_3x3_bn = mx.symbol.BatchNorm(name='reduction_a_3x3_bn', data=reduction_a_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_a_3x3_scale = reduction_a_3x3_bn
reduction_a_3x3_relu = mx.symbol.Activation(name='reduction_a_3x3_relu', data=reduction_a_3x3_scale , act_type='relu')
reduction_a_3x3_2_reduce = mx.symbol.Convolution(name='reduction_a_3x3_2_reduce', data=inception_resnet_v2_a10_residual_eltwise_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduction_a_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='reduction_a_3x3_2_reduce_bn', data=reduction_a_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_a_3x3_2_reduce_scale = reduction_a_3x3_2_reduce_bn
reduction_a_3x3_2_reduce_relu = mx.symbol.Activation(name='reduction_a_3x3_2_reduce_relu', data=reduction_a_3x3_2_reduce_scale , act_type='relu')
reduction_a_3x3_2 = mx.symbol.Convolution(name='reduction_a_3x3_2', data=reduction_a_3x3_2_reduce_relu , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
reduction_a_3x3_2_bn = mx.symbol.BatchNorm(name='reduction_a_3x3_2_bn', data=reduction_a_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_a_3x3_2_scale = reduction_a_3x3_2_bn
reduction_a_3x3_2_relu = mx.symbol.Activation(name='reduction_a_3x3_2_relu', data=reduction_a_3x3_2_scale , act_type='relu')
reduction_a_3x3_3 = mx.symbol.Convolution(name='reduction_a_3x3_3', data=reduction_a_3x3_2_relu , num_filter=384, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
reduction_a_3x3_3_bn = mx.symbol.BatchNorm(name='reduction_a_3x3_3_bn', data=reduction_a_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_a_3x3_3_scale = reduction_a_3x3_3_bn
reduction_a_3x3_3_relu = mx.symbol.Activation(name='reduction_a_3x3_3_relu', data=reduction_a_3x3_3_scale , act_type='relu')
reduction_a_pool = mx.symbol.Pooling(name='reduction_a_pool', data=inception_resnet_v2_a10_residual_eltwise_relu , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
reduction_a_concat = mx.symbol.Concat(name='reduction_a_concat', *[reduction_a_3x3_relu,reduction_a_3x3_3_relu,reduction_a_pool] )
inception_resnet_v2_b1_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b1_1x1', data=reduction_a_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b1_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b1_1x1_bn', data=inception_resnet_v2_b1_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b1_1x1_scale = inception_resnet_v2_b1_1x1_bn
inception_resnet_v2_b1_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b1_1x1_relu', data=inception_resnet_v2_b1_1x1_scale , act_type='relu')
inception_resnet_v2_b1_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b1_1x7_reduce', data=reduction_a_concat , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b1_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b1_1x7_reduce_bn', data=inception_resnet_v2_b1_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b1_1x7_reduce_scale = inception_resnet_v2_b1_1x7_reduce_bn
inception_resnet_v2_b1_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b1_1x7_reduce_relu', data=inception_resnet_v2_b1_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b1_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b1_1x7', data=inception_resnet_v2_b1_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b1_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b1_1x7_bn', data=inception_resnet_v2_b1_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b1_1x7_scale = inception_resnet_v2_b1_1x7_bn
inception_resnet_v2_b1_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b1_1x7_relu', data=inception_resnet_v2_b1_1x7_scale , act_type='relu')
inception_resnet_v2_b1_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b1_7x1', data=inception_resnet_v2_b1_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b1_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b1_7x1_bn', data=inception_resnet_v2_b1_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b1_7x1_scale = inception_resnet_v2_b1_7x1_bn
inception_resnet_v2_b1_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b1_7x1_relu', data=inception_resnet_v2_b1_7x1_scale , act_type='relu')
inception_resnet_v2_b1_concat = mx.symbol.Concat(name='inception_resnet_v2_b1_concat', *[inception_resnet_v2_b1_1x1_relu,inception_resnet_v2_b1_7x1_relu] )
inception_resnet_v2_b1_up = mx.symbol.Convolution(name='inception_resnet_v2_b1_up', data=inception_resnet_v2_b1_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b1_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b1_residual_eltwise', *[1.0*reduction_a_concat,0.10000000149*inception_resnet_v2_b1_up] )
inception_resnet_v2_b1_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b1_residual_eltwise_relu', data=inception_resnet_v2_b1_residual_eltwise , act_type='relu')
inception_resnet_v2_b2_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b2_1x1', data=inception_resnet_v2_b1_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b2_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b2_1x1_bn', data=inception_resnet_v2_b2_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b2_1x1_scale = inception_resnet_v2_b2_1x1_bn
inception_resnet_v2_b2_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b2_1x1_relu', data=inception_resnet_v2_b2_1x1_scale , act_type='relu')
inception_resnet_v2_b2_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b2_1x7_reduce', data=inception_resnet_v2_b1_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b2_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b2_1x7_reduce_bn', data=inception_resnet_v2_b2_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b2_1x7_reduce_scale = inception_resnet_v2_b2_1x7_reduce_bn
inception_resnet_v2_b2_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b2_1x7_reduce_relu', data=inception_resnet_v2_b2_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b2_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b2_1x7', data=inception_resnet_v2_b2_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b2_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b2_1x7_bn', data=inception_resnet_v2_b2_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b2_1x7_scale = inception_resnet_v2_b2_1x7_bn
inception_resnet_v2_b2_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b2_1x7_relu', data=inception_resnet_v2_b2_1x7_scale , act_type='relu')
inception_resnet_v2_b2_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b2_7x1', data=inception_resnet_v2_b2_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b2_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b2_7x1_bn', data=inception_resnet_v2_b2_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b2_7x1_scale = inception_resnet_v2_b2_7x1_bn
inception_resnet_v2_b2_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b2_7x1_relu', data=inception_resnet_v2_b2_7x1_scale , act_type='relu')
inception_resnet_v2_b2_concat = mx.symbol.Concat(name='inception_resnet_v2_b2_concat', *[inception_resnet_v2_b2_1x1_relu,inception_resnet_v2_b2_7x1_relu] )
inception_resnet_v2_b2_up = mx.symbol.Convolution(name='inception_resnet_v2_b2_up', data=inception_resnet_v2_b2_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b2_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b2_residual_eltwise', *[1.0*inception_resnet_v2_b1_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b2_up] )
inception_resnet_v2_b2_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b2_residual_eltwise_relu', data=inception_resnet_v2_b2_residual_eltwise , act_type='relu')
inception_resnet_v2_b3_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b3_1x1', data=inception_resnet_v2_b2_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b3_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b3_1x1_bn', data=inception_resnet_v2_b3_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b3_1x1_scale = inception_resnet_v2_b3_1x1_bn
inception_resnet_v2_b3_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b3_1x1_relu', data=inception_resnet_v2_b3_1x1_scale , act_type='relu')
inception_resnet_v2_b3_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b3_1x7_reduce', data=inception_resnet_v2_b2_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b3_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b3_1x7_reduce_bn', data=inception_resnet_v2_b3_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b3_1x7_reduce_scale = inception_resnet_v2_b3_1x7_reduce_bn
inception_resnet_v2_b3_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b3_1x7_reduce_relu', data=inception_resnet_v2_b3_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b3_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b3_1x7', data=inception_resnet_v2_b3_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b3_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b3_1x7_bn', data=inception_resnet_v2_b3_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b3_1x7_scale = inception_resnet_v2_b3_1x7_bn
inception_resnet_v2_b3_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b3_1x7_relu', data=inception_resnet_v2_b3_1x7_scale , act_type='relu')
inception_resnet_v2_b3_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b3_7x1', data=inception_resnet_v2_b3_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b3_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b3_7x1_bn', data=inception_resnet_v2_b3_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b3_7x1_scale = inception_resnet_v2_b3_7x1_bn
inception_resnet_v2_b3_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b3_7x1_relu', data=inception_resnet_v2_b3_7x1_scale , act_type='relu')
inception_resnet_v2_b3_concat = mx.symbol.Concat(name='inception_resnet_v2_b3_concat', *[inception_resnet_v2_b3_1x1_relu,inception_resnet_v2_b3_7x1_relu] )
inception_resnet_v2_b3_up = mx.symbol.Convolution(name='inception_resnet_v2_b3_up', data=inception_resnet_v2_b3_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b3_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b3_residual_eltwise', *[1.0*inception_resnet_v2_b2_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b3_up] )
inception_resnet_v2_b3_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b3_residual_eltwise_relu', data=inception_resnet_v2_b3_residual_eltwise , act_type='relu')
inception_resnet_v2_b4_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b4_1x1', data=inception_resnet_v2_b3_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b4_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b4_1x1_bn', data=inception_resnet_v2_b4_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b4_1x1_scale = inception_resnet_v2_b4_1x1_bn
inception_resnet_v2_b4_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b4_1x1_relu', data=inception_resnet_v2_b4_1x1_scale , act_type='relu')
inception_resnet_v2_b4_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b4_1x7_reduce', data=inception_resnet_v2_b3_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b4_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b4_1x7_reduce_bn', data=inception_resnet_v2_b4_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b4_1x7_reduce_scale = inception_resnet_v2_b4_1x7_reduce_bn
inception_resnet_v2_b4_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b4_1x7_reduce_relu', data=inception_resnet_v2_b4_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b4_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b4_1x7', data=inception_resnet_v2_b4_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b4_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b4_1x7_bn', data=inception_resnet_v2_b4_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b4_1x7_scale = inception_resnet_v2_b4_1x7_bn
inception_resnet_v2_b4_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b4_1x7_relu', data=inception_resnet_v2_b4_1x7_scale , act_type='relu')
inception_resnet_v2_b4_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b4_7x1', data=inception_resnet_v2_b4_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b4_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b4_7x1_bn', data=inception_resnet_v2_b4_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b4_7x1_scale = inception_resnet_v2_b4_7x1_bn
inception_resnet_v2_b4_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b4_7x1_relu', data=inception_resnet_v2_b4_7x1_scale , act_type='relu')
inception_resnet_v2_b4_concat = mx.symbol.Concat(name='inception_resnet_v2_b4_concat', *[inception_resnet_v2_b4_1x1_relu,inception_resnet_v2_b4_7x1_relu] )
inception_resnet_v2_b4_up = mx.symbol.Convolution(name='inception_resnet_v2_b4_up', data=inception_resnet_v2_b4_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b4_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b4_residual_eltwise', *[1.0*inception_resnet_v2_b3_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b4_up] )
inception_resnet_v2_b4_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b4_residual_eltwise_relu', data=inception_resnet_v2_b4_residual_eltwise , act_type='relu')
inception_resnet_v2_b5_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b5_1x1', data=inception_resnet_v2_b4_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b5_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b5_1x1_bn', data=inception_resnet_v2_b5_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b5_1x1_scale = inception_resnet_v2_b5_1x1_bn
inception_resnet_v2_b5_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b5_1x1_relu', data=inception_resnet_v2_b5_1x1_scale , act_type='relu')
inception_resnet_v2_b5_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b5_1x7_reduce', data=inception_resnet_v2_b4_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b5_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b5_1x7_reduce_bn', data=inception_resnet_v2_b5_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b5_1x7_reduce_scale = inception_resnet_v2_b5_1x7_reduce_bn
inception_resnet_v2_b5_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b5_1x7_reduce_relu', data=inception_resnet_v2_b5_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b5_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b5_1x7', data=inception_resnet_v2_b5_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b5_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b5_1x7_bn', data=inception_resnet_v2_b5_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b5_1x7_scale = inception_resnet_v2_b5_1x7_bn
inception_resnet_v2_b5_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b5_1x7_relu', data=inception_resnet_v2_b5_1x7_scale , act_type='relu')
inception_resnet_v2_b5_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b5_7x1', data=inception_resnet_v2_b5_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b5_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b5_7x1_bn', data=inception_resnet_v2_b5_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b5_7x1_scale = inception_resnet_v2_b5_7x1_bn
inception_resnet_v2_b5_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b5_7x1_relu', data=inception_resnet_v2_b5_7x1_scale , act_type='relu')
inception_resnet_v2_b5_concat = mx.symbol.Concat(name='inception_resnet_v2_b5_concat', *[inception_resnet_v2_b5_1x1_relu,inception_resnet_v2_b5_7x1_relu] )
inception_resnet_v2_b5_up = mx.symbol.Convolution(name='inception_resnet_v2_b5_up', data=inception_resnet_v2_b5_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b5_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b5_residual_eltwise', *[1.0*inception_resnet_v2_b4_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b5_up] )
inception_resnet_v2_b5_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b5_residual_eltwise_relu', data=inception_resnet_v2_b5_residual_eltwise , act_type='relu')
inception_resnet_v2_b6_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b6_1x1', data=inception_resnet_v2_b5_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b6_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b6_1x1_bn', data=inception_resnet_v2_b6_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b6_1x1_scale = inception_resnet_v2_b6_1x1_bn
inception_resnet_v2_b6_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b6_1x1_relu', data=inception_resnet_v2_b6_1x1_scale , act_type='relu')
inception_resnet_v2_b6_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b6_1x7_reduce', data=inception_resnet_v2_b5_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b6_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b6_1x7_reduce_bn', data=inception_resnet_v2_b6_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b6_1x7_reduce_scale = inception_resnet_v2_b6_1x7_reduce_bn
inception_resnet_v2_b6_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b6_1x7_reduce_relu', data=inception_resnet_v2_b6_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b6_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b6_1x7', data=inception_resnet_v2_b6_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b6_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b6_1x7_bn', data=inception_resnet_v2_b6_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b6_1x7_scale = inception_resnet_v2_b6_1x7_bn
inception_resnet_v2_b6_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b6_1x7_relu', data=inception_resnet_v2_b6_1x7_scale , act_type='relu')
inception_resnet_v2_b6_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b6_7x1', data=inception_resnet_v2_b6_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b6_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b6_7x1_bn', data=inception_resnet_v2_b6_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b6_7x1_scale = inception_resnet_v2_b6_7x1_bn
inception_resnet_v2_b6_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b6_7x1_relu', data=inception_resnet_v2_b6_7x1_scale , act_type='relu')
inception_resnet_v2_b6_concat = mx.symbol.Concat(name='inception_resnet_v2_b6_concat', *[inception_resnet_v2_b6_1x1_relu,inception_resnet_v2_b6_7x1_relu] )
inception_resnet_v2_b6_up = mx.symbol.Convolution(name='inception_resnet_v2_b6_up', data=inception_resnet_v2_b6_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b6_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b6_residual_eltwise', *[1.0*inception_resnet_v2_b5_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b6_up] )
inception_resnet_v2_b6_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b6_residual_eltwise_relu', data=inception_resnet_v2_b6_residual_eltwise , act_type='relu')
inception_resnet_v2_b7_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b7_1x1', data=inception_resnet_v2_b6_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b7_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b7_1x1_bn', data=inception_resnet_v2_b7_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b7_1x1_scale = inception_resnet_v2_b7_1x1_bn
inception_resnet_v2_b7_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b7_1x1_relu', data=inception_resnet_v2_b7_1x1_scale , act_type='relu')
inception_resnet_v2_b7_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b7_1x7_reduce', data=inception_resnet_v2_b6_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b7_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b7_1x7_reduce_bn', data=inception_resnet_v2_b7_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b7_1x7_reduce_scale = inception_resnet_v2_b7_1x7_reduce_bn
inception_resnet_v2_b7_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b7_1x7_reduce_relu', data=inception_resnet_v2_b7_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b7_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b7_1x7', data=inception_resnet_v2_b7_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b7_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b7_1x7_bn', data=inception_resnet_v2_b7_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b7_1x7_scale = inception_resnet_v2_b7_1x7_bn
inception_resnet_v2_b7_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b7_1x7_relu', data=inception_resnet_v2_b7_1x7_scale , act_type='relu')
inception_resnet_v2_b7_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b7_7x1', data=inception_resnet_v2_b7_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b7_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b7_7x1_bn', data=inception_resnet_v2_b7_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b7_7x1_scale = inception_resnet_v2_b7_7x1_bn
inception_resnet_v2_b7_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b7_7x1_relu', data=inception_resnet_v2_b7_7x1_scale , act_type='relu')
inception_resnet_v2_b7_concat = mx.symbol.Concat(name='inception_resnet_v2_b7_concat', *[inception_resnet_v2_b7_1x1_relu,inception_resnet_v2_b7_7x1_relu] )
inception_resnet_v2_b7_up = mx.symbol.Convolution(name='inception_resnet_v2_b7_up', data=inception_resnet_v2_b7_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b7_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b7_residual_eltwise', *[1.0*inception_resnet_v2_b6_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b7_up] )
inception_resnet_v2_b7_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b7_residual_eltwise_relu', data=inception_resnet_v2_b7_residual_eltwise , act_type='relu')
inception_resnet_v2_b8_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b8_1x1', data=inception_resnet_v2_b7_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b8_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b8_1x1_bn', data=inception_resnet_v2_b8_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b8_1x1_scale = inception_resnet_v2_b8_1x1_bn
inception_resnet_v2_b8_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b8_1x1_relu', data=inception_resnet_v2_b8_1x1_scale , act_type='relu')
inception_resnet_v2_b8_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b8_1x7_reduce', data=inception_resnet_v2_b7_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b8_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b8_1x7_reduce_bn', data=inception_resnet_v2_b8_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b8_1x7_reduce_scale = inception_resnet_v2_b8_1x7_reduce_bn
inception_resnet_v2_b8_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b8_1x7_reduce_relu', data=inception_resnet_v2_b8_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b8_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b8_1x7', data=inception_resnet_v2_b8_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b8_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b8_1x7_bn', data=inception_resnet_v2_b8_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b8_1x7_scale = inception_resnet_v2_b8_1x7_bn
inception_resnet_v2_b8_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b8_1x7_relu', data=inception_resnet_v2_b8_1x7_scale , act_type='relu')
inception_resnet_v2_b8_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b8_7x1', data=inception_resnet_v2_b8_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b8_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b8_7x1_bn', data=inception_resnet_v2_b8_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b8_7x1_scale = inception_resnet_v2_b8_7x1_bn
inception_resnet_v2_b8_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b8_7x1_relu', data=inception_resnet_v2_b8_7x1_scale , act_type='relu')
inception_resnet_v2_b8_concat = mx.symbol.Concat(name='inception_resnet_v2_b8_concat', *[inception_resnet_v2_b8_1x1_relu,inception_resnet_v2_b8_7x1_relu] )
inception_resnet_v2_b8_up = mx.symbol.Convolution(name='inception_resnet_v2_b8_up', data=inception_resnet_v2_b8_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b8_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b8_residual_eltwise', *[1.0*inception_resnet_v2_b7_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b8_up] )
inception_resnet_v2_b8_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b8_residual_eltwise_relu', data=inception_resnet_v2_b8_residual_eltwise , act_type='relu')
inception_resnet_v2_b9_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b9_1x1', data=inception_resnet_v2_b8_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b9_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b9_1x1_bn', data=inception_resnet_v2_b9_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b9_1x1_scale = inception_resnet_v2_b9_1x1_bn
inception_resnet_v2_b9_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b9_1x1_relu', data=inception_resnet_v2_b9_1x1_scale , act_type='relu')
inception_resnet_v2_b9_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b9_1x7_reduce', data=inception_resnet_v2_b8_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b9_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b9_1x7_reduce_bn', data=inception_resnet_v2_b9_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b9_1x7_reduce_scale = inception_resnet_v2_b9_1x7_reduce_bn
inception_resnet_v2_b9_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b9_1x7_reduce_relu', data=inception_resnet_v2_b9_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b9_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b9_1x7', data=inception_resnet_v2_b9_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b9_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b9_1x7_bn', data=inception_resnet_v2_b9_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b9_1x7_scale = inception_resnet_v2_b9_1x7_bn
inception_resnet_v2_b9_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b9_1x7_relu', data=inception_resnet_v2_b9_1x7_scale , act_type='relu')
inception_resnet_v2_b9_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b9_7x1', data=inception_resnet_v2_b9_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b9_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b9_7x1_bn', data=inception_resnet_v2_b9_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b9_7x1_scale = inception_resnet_v2_b9_7x1_bn
inception_resnet_v2_b9_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b9_7x1_relu', data=inception_resnet_v2_b9_7x1_scale , act_type='relu')
inception_resnet_v2_b9_concat = mx.symbol.Concat(name='inception_resnet_v2_b9_concat', *[inception_resnet_v2_b9_1x1_relu,inception_resnet_v2_b9_7x1_relu] )
inception_resnet_v2_b9_up = mx.symbol.Convolution(name='inception_resnet_v2_b9_up', data=inception_resnet_v2_b9_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b9_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b9_residual_eltwise', *[1.0*inception_resnet_v2_b8_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b9_up] )
inception_resnet_v2_b9_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b9_residual_eltwise_relu', data=inception_resnet_v2_b9_residual_eltwise , act_type='relu')
inception_resnet_v2_b10_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b10_1x1', data=inception_resnet_v2_b9_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b10_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b10_1x1_bn', data=inception_resnet_v2_b10_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b10_1x1_scale = inception_resnet_v2_b10_1x1_bn
inception_resnet_v2_b10_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b10_1x1_relu', data=inception_resnet_v2_b10_1x1_scale , act_type='relu')
inception_resnet_v2_b10_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b10_1x7_reduce', data=inception_resnet_v2_b9_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b10_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b10_1x7_reduce_bn', data=inception_resnet_v2_b10_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b10_1x7_reduce_scale = inception_resnet_v2_b10_1x7_reduce_bn
inception_resnet_v2_b10_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b10_1x7_reduce_relu', data=inception_resnet_v2_b10_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b10_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b10_1x7', data=inception_resnet_v2_b10_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b10_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b10_1x7_bn', data=inception_resnet_v2_b10_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b10_1x7_scale = inception_resnet_v2_b10_1x7_bn
inception_resnet_v2_b10_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b10_1x7_relu', data=inception_resnet_v2_b10_1x7_scale , act_type='relu')
inception_resnet_v2_b10_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b10_7x1', data=inception_resnet_v2_b10_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b10_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b10_7x1_bn', data=inception_resnet_v2_b10_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b10_7x1_scale = inception_resnet_v2_b10_7x1_bn
inception_resnet_v2_b10_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b10_7x1_relu', data=inception_resnet_v2_b10_7x1_scale , act_type='relu')
inception_resnet_v2_b10_concat = mx.symbol.Concat(name='inception_resnet_v2_b10_concat', *[inception_resnet_v2_b10_1x1_relu,inception_resnet_v2_b10_7x1_relu] )
inception_resnet_v2_b10_up = mx.symbol.Convolution(name='inception_resnet_v2_b10_up', data=inception_resnet_v2_b10_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b10_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b10_residual_eltwise', *[1.0*inception_resnet_v2_b9_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b10_up] )
inception_resnet_v2_b10_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b10_residual_eltwise_relu', data=inception_resnet_v2_b10_residual_eltwise , act_type='relu')
inception_resnet_v2_b11_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b11_1x1', data=inception_resnet_v2_b10_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b11_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b11_1x1_bn', data=inception_resnet_v2_b11_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b11_1x1_scale = inception_resnet_v2_b11_1x1_bn
inception_resnet_v2_b11_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b11_1x1_relu', data=inception_resnet_v2_b11_1x1_scale , act_type='relu')
inception_resnet_v2_b11_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b11_1x7_reduce', data=inception_resnet_v2_b10_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b11_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b11_1x7_reduce_bn', data=inception_resnet_v2_b11_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b11_1x7_reduce_scale = inception_resnet_v2_b11_1x7_reduce_bn
inception_resnet_v2_b11_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b11_1x7_reduce_relu', data=inception_resnet_v2_b11_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b11_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b11_1x7', data=inception_resnet_v2_b11_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b11_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b11_1x7_bn', data=inception_resnet_v2_b11_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b11_1x7_scale = inception_resnet_v2_b11_1x7_bn
inception_resnet_v2_b11_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b11_1x7_relu', data=inception_resnet_v2_b11_1x7_scale , act_type='relu')
inception_resnet_v2_b11_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b11_7x1', data=inception_resnet_v2_b11_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b11_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b11_7x1_bn', data=inception_resnet_v2_b11_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b11_7x1_scale = inception_resnet_v2_b11_7x1_bn
inception_resnet_v2_b11_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b11_7x1_relu', data=inception_resnet_v2_b11_7x1_scale , act_type='relu')
inception_resnet_v2_b11_concat = mx.symbol.Concat(name='inception_resnet_v2_b11_concat', *[inception_resnet_v2_b11_1x1_relu,inception_resnet_v2_b11_7x1_relu] )
inception_resnet_v2_b11_up = mx.symbol.Convolution(name='inception_resnet_v2_b11_up', data=inception_resnet_v2_b11_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b11_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b11_residual_eltwise', *[1.0*inception_resnet_v2_b10_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b11_up] )
inception_resnet_v2_b11_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b11_residual_eltwise_relu', data=inception_resnet_v2_b11_residual_eltwise , act_type='relu')
inception_resnet_v2_b12_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b12_1x1', data=inception_resnet_v2_b11_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b12_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b12_1x1_bn', data=inception_resnet_v2_b12_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b12_1x1_scale = inception_resnet_v2_b12_1x1_bn
inception_resnet_v2_b12_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b12_1x1_relu', data=inception_resnet_v2_b12_1x1_scale , act_type='relu')
inception_resnet_v2_b12_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b12_1x7_reduce', data=inception_resnet_v2_b11_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b12_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b12_1x7_reduce_bn', data=inception_resnet_v2_b12_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b12_1x7_reduce_scale = inception_resnet_v2_b12_1x7_reduce_bn
inception_resnet_v2_b12_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b12_1x7_reduce_relu', data=inception_resnet_v2_b12_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b12_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b12_1x7', data=inception_resnet_v2_b12_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b12_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b12_1x7_bn', data=inception_resnet_v2_b12_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b12_1x7_scale = inception_resnet_v2_b12_1x7_bn
inception_resnet_v2_b12_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b12_1x7_relu', data=inception_resnet_v2_b12_1x7_scale , act_type='relu')
inception_resnet_v2_b12_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b12_7x1', data=inception_resnet_v2_b12_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b12_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b12_7x1_bn', data=inception_resnet_v2_b12_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b12_7x1_scale = inception_resnet_v2_b12_7x1_bn
inception_resnet_v2_b12_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b12_7x1_relu', data=inception_resnet_v2_b12_7x1_scale , act_type='relu')
inception_resnet_v2_b12_concat = mx.symbol.Concat(name='inception_resnet_v2_b12_concat', *[inception_resnet_v2_b12_1x1_relu,inception_resnet_v2_b12_7x1_relu] )
inception_resnet_v2_b12_up = mx.symbol.Convolution(name='inception_resnet_v2_b12_up', data=inception_resnet_v2_b12_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b12_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b12_residual_eltwise', *[1.0*inception_resnet_v2_b11_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b12_up] )
inception_resnet_v2_b12_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b12_residual_eltwise_relu', data=inception_resnet_v2_b12_residual_eltwise , act_type='relu')
inception_resnet_v2_b13_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b13_1x1', data=inception_resnet_v2_b12_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b13_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b13_1x1_bn', data=inception_resnet_v2_b13_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b13_1x1_scale = inception_resnet_v2_b13_1x1_bn
inception_resnet_v2_b13_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b13_1x1_relu', data=inception_resnet_v2_b13_1x1_scale , act_type='relu')
inception_resnet_v2_b13_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b13_1x7_reduce', data=inception_resnet_v2_b12_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b13_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b13_1x7_reduce_bn', data=inception_resnet_v2_b13_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b13_1x7_reduce_scale = inception_resnet_v2_b13_1x7_reduce_bn
inception_resnet_v2_b13_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b13_1x7_reduce_relu', data=inception_resnet_v2_b13_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b13_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b13_1x7', data=inception_resnet_v2_b13_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b13_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b13_1x7_bn', data=inception_resnet_v2_b13_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b13_1x7_scale = inception_resnet_v2_b13_1x7_bn
inception_resnet_v2_b13_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b13_1x7_relu', data=inception_resnet_v2_b13_1x7_scale , act_type='relu')
inception_resnet_v2_b13_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b13_7x1', data=inception_resnet_v2_b13_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b13_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b13_7x1_bn', data=inception_resnet_v2_b13_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b13_7x1_scale = inception_resnet_v2_b13_7x1_bn
inception_resnet_v2_b13_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b13_7x1_relu', data=inception_resnet_v2_b13_7x1_scale , act_type='relu')
inception_resnet_v2_b13_concat = mx.symbol.Concat(name='inception_resnet_v2_b13_concat', *[inception_resnet_v2_b13_1x1_relu,inception_resnet_v2_b13_7x1_relu] )
inception_resnet_v2_b13_up = mx.symbol.Convolution(name='inception_resnet_v2_b13_up', data=inception_resnet_v2_b13_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b13_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b13_residual_eltwise', *[1.0*inception_resnet_v2_b12_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b13_up] )
inception_resnet_v2_b13_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b13_residual_eltwise_relu', data=inception_resnet_v2_b13_residual_eltwise , act_type='relu')
inception_resnet_v2_b14_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b14_1x1', data=inception_resnet_v2_b13_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b14_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b14_1x1_bn', data=inception_resnet_v2_b14_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b14_1x1_scale = inception_resnet_v2_b14_1x1_bn
inception_resnet_v2_b14_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b14_1x1_relu', data=inception_resnet_v2_b14_1x1_scale , act_type='relu')
inception_resnet_v2_b14_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b14_1x7_reduce', data=inception_resnet_v2_b13_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b14_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b14_1x7_reduce_bn', data=inception_resnet_v2_b14_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b14_1x7_reduce_scale = inception_resnet_v2_b14_1x7_reduce_bn
inception_resnet_v2_b14_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b14_1x7_reduce_relu', data=inception_resnet_v2_b14_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b14_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b14_1x7', data=inception_resnet_v2_b14_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b14_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b14_1x7_bn', data=inception_resnet_v2_b14_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b14_1x7_scale = inception_resnet_v2_b14_1x7_bn
inception_resnet_v2_b14_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b14_1x7_relu', data=inception_resnet_v2_b14_1x7_scale , act_type='relu')
inception_resnet_v2_b14_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b14_7x1', data=inception_resnet_v2_b14_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b14_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b14_7x1_bn', data=inception_resnet_v2_b14_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b14_7x1_scale = inception_resnet_v2_b14_7x1_bn
inception_resnet_v2_b14_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b14_7x1_relu', data=inception_resnet_v2_b14_7x1_scale , act_type='relu')
inception_resnet_v2_b14_concat = mx.symbol.Concat(name='inception_resnet_v2_b14_concat', *[inception_resnet_v2_b14_1x1_relu,inception_resnet_v2_b14_7x1_relu] )
inception_resnet_v2_b14_up = mx.symbol.Convolution(name='inception_resnet_v2_b14_up', data=inception_resnet_v2_b14_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b14_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b14_residual_eltwise', *[1.0*inception_resnet_v2_b13_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b14_up] )
inception_resnet_v2_b14_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b14_residual_eltwise_relu', data=inception_resnet_v2_b14_residual_eltwise , act_type='relu')
inception_resnet_v2_b15_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b15_1x1', data=inception_resnet_v2_b14_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b15_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b15_1x1_bn', data=inception_resnet_v2_b15_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b15_1x1_scale = inception_resnet_v2_b15_1x1_bn
inception_resnet_v2_b15_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b15_1x1_relu', data=inception_resnet_v2_b15_1x1_scale , act_type='relu')
inception_resnet_v2_b15_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b15_1x7_reduce', data=inception_resnet_v2_b14_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b15_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b15_1x7_reduce_bn', data=inception_resnet_v2_b15_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b15_1x7_reduce_scale = inception_resnet_v2_b15_1x7_reduce_bn
inception_resnet_v2_b15_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b15_1x7_reduce_relu', data=inception_resnet_v2_b15_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b15_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b15_1x7', data=inception_resnet_v2_b15_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b15_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b15_1x7_bn', data=inception_resnet_v2_b15_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b15_1x7_scale = inception_resnet_v2_b15_1x7_bn
inception_resnet_v2_b15_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b15_1x7_relu', data=inception_resnet_v2_b15_1x7_scale , act_type='relu')
inception_resnet_v2_b15_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b15_7x1', data=inception_resnet_v2_b15_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b15_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b15_7x1_bn', data=inception_resnet_v2_b15_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b15_7x1_scale = inception_resnet_v2_b15_7x1_bn
inception_resnet_v2_b15_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b15_7x1_relu', data=inception_resnet_v2_b15_7x1_scale , act_type='relu')
inception_resnet_v2_b15_concat = mx.symbol.Concat(name='inception_resnet_v2_b15_concat', *[inception_resnet_v2_b15_1x1_relu,inception_resnet_v2_b15_7x1_relu] )
inception_resnet_v2_b15_up = mx.symbol.Convolution(name='inception_resnet_v2_b15_up', data=inception_resnet_v2_b15_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b15_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b15_residual_eltwise', *[1.0*inception_resnet_v2_b14_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b15_up] )
inception_resnet_v2_b15_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b15_residual_eltwise_relu', data=inception_resnet_v2_b15_residual_eltwise , act_type='relu')
inception_resnet_v2_b16_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b16_1x1', data=inception_resnet_v2_b15_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b16_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b16_1x1_bn', data=inception_resnet_v2_b16_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b16_1x1_scale = inception_resnet_v2_b16_1x1_bn
inception_resnet_v2_b16_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b16_1x1_relu', data=inception_resnet_v2_b16_1x1_scale , act_type='relu')
inception_resnet_v2_b16_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b16_1x7_reduce', data=inception_resnet_v2_b15_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b16_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b16_1x7_reduce_bn', data=inception_resnet_v2_b16_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b16_1x7_reduce_scale = inception_resnet_v2_b16_1x7_reduce_bn
inception_resnet_v2_b16_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b16_1x7_reduce_relu', data=inception_resnet_v2_b16_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b16_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b16_1x7', data=inception_resnet_v2_b16_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b16_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b16_1x7_bn', data=inception_resnet_v2_b16_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b16_1x7_scale = inception_resnet_v2_b16_1x7_bn
inception_resnet_v2_b16_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b16_1x7_relu', data=inception_resnet_v2_b16_1x7_scale , act_type='relu')
inception_resnet_v2_b16_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b16_7x1', data=inception_resnet_v2_b16_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b16_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b16_7x1_bn', data=inception_resnet_v2_b16_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b16_7x1_scale = inception_resnet_v2_b16_7x1_bn
inception_resnet_v2_b16_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b16_7x1_relu', data=inception_resnet_v2_b16_7x1_scale , act_type='relu')
inception_resnet_v2_b16_concat = mx.symbol.Concat(name='inception_resnet_v2_b16_concat', *[inception_resnet_v2_b16_1x1_relu,inception_resnet_v2_b16_7x1_relu] )
inception_resnet_v2_b16_up = mx.symbol.Convolution(name='inception_resnet_v2_b16_up', data=inception_resnet_v2_b16_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b16_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b16_residual_eltwise', *[1.0*inception_resnet_v2_b15_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b16_up] )
inception_resnet_v2_b16_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b16_residual_eltwise_relu', data=inception_resnet_v2_b16_residual_eltwise , act_type='relu')
inception_resnet_v2_b17_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b17_1x1', data=inception_resnet_v2_b16_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b17_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b17_1x1_bn', data=inception_resnet_v2_b17_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b17_1x1_scale = inception_resnet_v2_b17_1x1_bn
inception_resnet_v2_b17_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b17_1x1_relu', data=inception_resnet_v2_b17_1x1_scale , act_type='relu')
inception_resnet_v2_b17_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b17_1x7_reduce', data=inception_resnet_v2_b16_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b17_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b17_1x7_reduce_bn', data=inception_resnet_v2_b17_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b17_1x7_reduce_scale = inception_resnet_v2_b17_1x7_reduce_bn
inception_resnet_v2_b17_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b17_1x7_reduce_relu', data=inception_resnet_v2_b17_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b17_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b17_1x7', data=inception_resnet_v2_b17_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b17_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b17_1x7_bn', data=inception_resnet_v2_b17_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b17_1x7_scale = inception_resnet_v2_b17_1x7_bn
inception_resnet_v2_b17_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b17_1x7_relu', data=inception_resnet_v2_b17_1x7_scale , act_type='relu')
inception_resnet_v2_b17_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b17_7x1', data=inception_resnet_v2_b17_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b17_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b17_7x1_bn', data=inception_resnet_v2_b17_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b17_7x1_scale = inception_resnet_v2_b17_7x1_bn
inception_resnet_v2_b17_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b17_7x1_relu', data=inception_resnet_v2_b17_7x1_scale , act_type='relu')
inception_resnet_v2_b17_concat = mx.symbol.Concat(name='inception_resnet_v2_b17_concat', *[inception_resnet_v2_b17_1x1_relu,inception_resnet_v2_b17_7x1_relu] )
inception_resnet_v2_b17_up = mx.symbol.Convolution(name='inception_resnet_v2_b17_up', data=inception_resnet_v2_b17_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b17_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b17_residual_eltwise', *[1.0*inception_resnet_v2_b16_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b17_up] )
inception_resnet_v2_b17_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b17_residual_eltwise_relu', data=inception_resnet_v2_b17_residual_eltwise , act_type='relu')
inception_resnet_v2_b18_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b18_1x1', data=inception_resnet_v2_b17_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b18_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b18_1x1_bn', data=inception_resnet_v2_b18_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b18_1x1_scale = inception_resnet_v2_b18_1x1_bn
inception_resnet_v2_b18_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b18_1x1_relu', data=inception_resnet_v2_b18_1x1_scale , act_type='relu')
inception_resnet_v2_b18_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b18_1x7_reduce', data=inception_resnet_v2_b17_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b18_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b18_1x7_reduce_bn', data=inception_resnet_v2_b18_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b18_1x7_reduce_scale = inception_resnet_v2_b18_1x7_reduce_bn
inception_resnet_v2_b18_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b18_1x7_reduce_relu', data=inception_resnet_v2_b18_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b18_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b18_1x7', data=inception_resnet_v2_b18_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b18_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b18_1x7_bn', data=inception_resnet_v2_b18_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b18_1x7_scale = inception_resnet_v2_b18_1x7_bn
inception_resnet_v2_b18_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b18_1x7_relu', data=inception_resnet_v2_b18_1x7_scale , act_type='relu')
inception_resnet_v2_b18_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b18_7x1', data=inception_resnet_v2_b18_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b18_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b18_7x1_bn', data=inception_resnet_v2_b18_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b18_7x1_scale = inception_resnet_v2_b18_7x1_bn
inception_resnet_v2_b18_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b18_7x1_relu', data=inception_resnet_v2_b18_7x1_scale , act_type='relu')
inception_resnet_v2_b18_concat = mx.symbol.Concat(name='inception_resnet_v2_b18_concat', *[inception_resnet_v2_b18_1x1_relu,inception_resnet_v2_b18_7x1_relu] )
inception_resnet_v2_b18_up = mx.symbol.Convolution(name='inception_resnet_v2_b18_up', data=inception_resnet_v2_b18_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b18_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b18_residual_eltwise', *[1.0*inception_resnet_v2_b17_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b18_up] )
inception_resnet_v2_b18_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b18_residual_eltwise_relu', data=inception_resnet_v2_b18_residual_eltwise , act_type='relu')
inception_resnet_v2_b19_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b19_1x1', data=inception_resnet_v2_b18_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b19_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b19_1x1_bn', data=inception_resnet_v2_b19_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b19_1x1_scale = inception_resnet_v2_b19_1x1_bn
inception_resnet_v2_b19_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b19_1x1_relu', data=inception_resnet_v2_b19_1x1_scale , act_type='relu')
inception_resnet_v2_b19_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b19_1x7_reduce', data=inception_resnet_v2_b18_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b19_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b19_1x7_reduce_bn', data=inception_resnet_v2_b19_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b19_1x7_reduce_scale = inception_resnet_v2_b19_1x7_reduce_bn
inception_resnet_v2_b19_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b19_1x7_reduce_relu', data=inception_resnet_v2_b19_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b19_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b19_1x7', data=inception_resnet_v2_b19_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b19_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b19_1x7_bn', data=inception_resnet_v2_b19_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b19_1x7_scale = inception_resnet_v2_b19_1x7_bn
inception_resnet_v2_b19_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b19_1x7_relu', data=inception_resnet_v2_b19_1x7_scale , act_type='relu')
inception_resnet_v2_b19_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b19_7x1', data=inception_resnet_v2_b19_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b19_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b19_7x1_bn', data=inception_resnet_v2_b19_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b19_7x1_scale = inception_resnet_v2_b19_7x1_bn
inception_resnet_v2_b19_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b19_7x1_relu', data=inception_resnet_v2_b19_7x1_scale , act_type='relu')
inception_resnet_v2_b19_concat = mx.symbol.Concat(name='inception_resnet_v2_b19_concat', *[inception_resnet_v2_b19_1x1_relu,inception_resnet_v2_b19_7x1_relu] )
inception_resnet_v2_b19_up = mx.symbol.Convolution(name='inception_resnet_v2_b19_up', data=inception_resnet_v2_b19_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b19_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b19_residual_eltwise', *[1.0*inception_resnet_v2_b18_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b19_up] )
inception_resnet_v2_b19_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b19_residual_eltwise_relu', data=inception_resnet_v2_b19_residual_eltwise , act_type='relu')
inception_resnet_v2_b20_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_b20_1x1', data=inception_resnet_v2_b19_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b20_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b20_1x1_bn', data=inception_resnet_v2_b20_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b20_1x1_scale = inception_resnet_v2_b20_1x1_bn
inception_resnet_v2_b20_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b20_1x1_relu', data=inception_resnet_v2_b20_1x1_scale , act_type='relu')
inception_resnet_v2_b20_1x7_reduce = mx.symbol.Convolution(name='inception_resnet_v2_b20_1x7_reduce', data=inception_resnet_v2_b19_residual_eltwise_relu , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b20_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b20_1x7_reduce_bn', data=inception_resnet_v2_b20_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b20_1x7_reduce_scale = inception_resnet_v2_b20_1x7_reduce_bn
inception_resnet_v2_b20_1x7_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_b20_1x7_reduce_relu', data=inception_resnet_v2_b20_1x7_reduce_scale , act_type='relu')
inception_resnet_v2_b20_1x7 = mx.symbol.Convolution(name='inception_resnet_v2_b20_1x7', data=inception_resnet_v2_b20_1x7_reduce_relu , num_filter=160, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_resnet_v2_b20_1x7_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b20_1x7_bn', data=inception_resnet_v2_b20_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b20_1x7_scale = inception_resnet_v2_b20_1x7_bn
inception_resnet_v2_b20_1x7_relu = mx.symbol.Activation(name='inception_resnet_v2_b20_1x7_relu', data=inception_resnet_v2_b20_1x7_scale , act_type='relu')
inception_resnet_v2_b20_7x1 = mx.symbol.Convolution(name='inception_resnet_v2_b20_7x1', data=inception_resnet_v2_b20_1x7_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_resnet_v2_b20_7x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_b20_7x1_bn', data=inception_resnet_v2_b20_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_b20_7x1_scale = inception_resnet_v2_b20_7x1_bn
inception_resnet_v2_b20_7x1_relu = mx.symbol.Activation(name='inception_resnet_v2_b20_7x1_relu', data=inception_resnet_v2_b20_7x1_scale , act_type='relu')
inception_resnet_v2_b20_concat = mx.symbol.Concat(name='inception_resnet_v2_b20_concat', *[inception_resnet_v2_b20_1x1_relu,inception_resnet_v2_b20_7x1_relu] )
inception_resnet_v2_b20_up = mx.symbol.Convolution(name='inception_resnet_v2_b20_up', data=inception_resnet_v2_b20_concat , num_filter=1088, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_b20_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_b20_residual_eltwise', *[1.0*inception_resnet_v2_b19_residual_eltwise_relu,0.10000000149*inception_resnet_v2_b20_up] )
inception_resnet_v2_b20_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_b20_residual_eltwise_relu', data=inception_resnet_v2_b20_residual_eltwise , act_type='relu')
reduction_b_3x3_reduce = mx.symbol.Convolution(name='reduction_b_3x3_reduce', data=inception_resnet_v2_b20_residual_eltwise_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduction_b_3x3_reduce_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_reduce_bn', data=reduction_b_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_reduce_scale = reduction_b_3x3_reduce_bn
reduction_b_3x3_reduce_relu = mx.symbol.Activation(name='reduction_b_3x3_reduce_relu', data=reduction_b_3x3_reduce_scale , act_type='relu')
reduction_b_3x3 = mx.symbol.Convolution(name='reduction_b_3x3', data=reduction_b_3x3_reduce_relu , num_filter=384, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
reduction_b_3x3_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_bn', data=reduction_b_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_scale = reduction_b_3x3_bn
reduction_b_3x3_relu = mx.symbol.Activation(name='reduction_b_3x3_relu', data=reduction_b_3x3_scale , act_type='relu')
reduction_b_3x3_2_reduce = mx.symbol.Convolution(name='reduction_b_3x3_2_reduce', data=inception_resnet_v2_b20_residual_eltwise_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduction_b_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_2_reduce_bn', data=reduction_b_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_2_reduce_scale = reduction_b_3x3_2_reduce_bn
reduction_b_3x3_2_reduce_relu = mx.symbol.Activation(name='reduction_b_3x3_2_reduce_relu', data=reduction_b_3x3_2_reduce_scale , act_type='relu')
reduction_b_3x3_2 = mx.symbol.Convolution(name='reduction_b_3x3_2', data=reduction_b_3x3_2_reduce_relu , num_filter=288, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
reduction_b_3x3_2_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_2_bn', data=reduction_b_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_2_scale = reduction_b_3x3_2_bn
reduction_b_3x3_2_relu = mx.symbol.Activation(name='reduction_b_3x3_2_relu', data=reduction_b_3x3_2_scale , act_type='relu')
reduction_b_3x3_3_reduce = mx.symbol.Convolution(name='reduction_b_3x3_3_reduce', data=inception_resnet_v2_b20_residual_eltwise_relu , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduction_b_3x3_3_reduce_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_3_reduce_bn', data=reduction_b_3x3_3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_3_reduce_scale = reduction_b_3x3_3_reduce_bn
reduction_b_3x3_3_reduce_relu = mx.symbol.Activation(name='reduction_b_3x3_3_reduce_relu', data=reduction_b_3x3_3_reduce_scale , act_type='relu')
reduction_b_3x3_3 = mx.symbol.Convolution(name='reduction_b_3x3_3', data=reduction_b_3x3_3_reduce_relu , num_filter=288, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
reduction_b_3x3_3_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_3_bn', data=reduction_b_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_3_scale = reduction_b_3x3_3_bn
reduction_b_3x3_3_relu = mx.symbol.Activation(name='reduction_b_3x3_3_relu', data=reduction_b_3x3_3_scale , act_type='relu')
reduction_b_3x3_4 = mx.symbol.Convolution(name='reduction_b_3x3_4', data=reduction_b_3x3_3_relu , num_filter=320, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
reduction_b_3x3_4_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_4_bn', data=reduction_b_3x3_4 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_4_scale = reduction_b_3x3_4_bn
reduction_b_3x3_4_relu = mx.symbol.Activation(name='reduction_b_3x3_4_relu', data=reduction_b_3x3_4_scale , act_type='relu')
reduction_b_pool = mx.symbol.Pooling(name='reduction_b_pool', data=inception_resnet_v2_b20_residual_eltwise_relu , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
reduction_b_concat = mx.symbol.Concat(name='reduction_b_concat', *[reduction_b_3x3_relu,reduction_b_3x3_2_relu,reduction_b_3x3_4_relu,reduction_b_pool] )
inception_resnet_v2_c1_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c1_1x1', data=reduction_b_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c1_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c1_1x1_bn', data=inception_resnet_v2_c1_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c1_1x1_scale = inception_resnet_v2_c1_1x1_bn
inception_resnet_v2_c1_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c1_1x1_relu', data=inception_resnet_v2_c1_1x1_scale , act_type='relu')
inception_resnet_v2_c1_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c1_1x3_reduce', data=reduction_b_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c1_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c1_1x3_reduce_bn', data=inception_resnet_v2_c1_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c1_1x3_reduce_scale = inception_resnet_v2_c1_1x3_reduce_bn
inception_resnet_v2_c1_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c1_1x3_reduce_relu', data=inception_resnet_v2_c1_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c1_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c1_1x3', data=inception_resnet_v2_c1_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c1_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c1_1x3_bn', data=inception_resnet_v2_c1_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c1_1x3_scale = inception_resnet_v2_c1_1x3_bn
inception_resnet_v2_c1_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c1_1x3_relu', data=inception_resnet_v2_c1_1x3_scale , act_type='relu')
inception_resnet_v2_c1_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c1_3x1', data=inception_resnet_v2_c1_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c1_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c1_3x1_bn', data=inception_resnet_v2_c1_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c1_3x1_scale = inception_resnet_v2_c1_3x1_bn
inception_resnet_v2_c1_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c1_3x1_relu', data=inception_resnet_v2_c1_3x1_scale , act_type='relu')
inception_resnet_v2_c1_concat = mx.symbol.Concat(name='inception_resnet_v2_c1_concat', *[inception_resnet_v2_c1_1x1_relu,inception_resnet_v2_c1_3x1_relu] )
inception_resnet_v2_c1_up = mx.symbol.Convolution(name='inception_resnet_v2_c1_up', data=inception_resnet_v2_c1_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c1_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c1_residual_eltwise', *[1.0*reduction_b_concat,0.20000000298*inception_resnet_v2_c1_up] )
inception_resnet_v2_c1_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_c1_residual_eltwise_relu', data=inception_resnet_v2_c1_residual_eltwise , act_type='relu')
inception_resnet_v2_c2_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c2_1x1', data=inception_resnet_v2_c1_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c2_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c2_1x1_bn', data=inception_resnet_v2_c2_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c2_1x1_scale = inception_resnet_v2_c2_1x1_bn
inception_resnet_v2_c2_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c2_1x1_relu', data=inception_resnet_v2_c2_1x1_scale , act_type='relu')
inception_resnet_v2_c2_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c2_1x3_reduce', data=inception_resnet_v2_c1_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c2_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c2_1x3_reduce_bn', data=inception_resnet_v2_c2_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c2_1x3_reduce_scale = inception_resnet_v2_c2_1x3_reduce_bn
inception_resnet_v2_c2_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c2_1x3_reduce_relu', data=inception_resnet_v2_c2_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c2_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c2_1x3', data=inception_resnet_v2_c2_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c2_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c2_1x3_bn', data=inception_resnet_v2_c2_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c2_1x3_scale = inception_resnet_v2_c2_1x3_bn
inception_resnet_v2_c2_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c2_1x3_relu', data=inception_resnet_v2_c2_1x3_scale , act_type='relu')
inception_resnet_v2_c2_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c2_3x1', data=inception_resnet_v2_c2_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c2_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c2_3x1_bn', data=inception_resnet_v2_c2_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c2_3x1_scale = inception_resnet_v2_c2_3x1_bn
inception_resnet_v2_c2_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c2_3x1_relu', data=inception_resnet_v2_c2_3x1_scale , act_type='relu')
inception_resnet_v2_c2_concat = mx.symbol.Concat(name='inception_resnet_v2_c2_concat', *[inception_resnet_v2_c2_1x1_relu,inception_resnet_v2_c2_3x1_relu] )
inception_resnet_v2_c2_up = mx.symbol.Convolution(name='inception_resnet_v2_c2_up', data=inception_resnet_v2_c2_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c2_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c2_residual_eltwise', *[1.0*inception_resnet_v2_c1_residual_eltwise_relu,0.20000000298*inception_resnet_v2_c2_up] )
inception_resnet_v2_c2_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_c2_residual_eltwise_relu', data=inception_resnet_v2_c2_residual_eltwise , act_type='relu')
inception_resnet_v2_c3_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c3_1x1', data=inception_resnet_v2_c2_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c3_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c3_1x1_bn', data=inception_resnet_v2_c3_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c3_1x1_scale = inception_resnet_v2_c3_1x1_bn
inception_resnet_v2_c3_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c3_1x1_relu', data=inception_resnet_v2_c3_1x1_scale , act_type='relu')
inception_resnet_v2_c3_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c3_1x3_reduce', data=inception_resnet_v2_c2_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c3_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c3_1x3_reduce_bn', data=inception_resnet_v2_c3_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c3_1x3_reduce_scale = inception_resnet_v2_c3_1x3_reduce_bn
inception_resnet_v2_c3_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c3_1x3_reduce_relu', data=inception_resnet_v2_c3_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c3_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c3_1x3', data=inception_resnet_v2_c3_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c3_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c3_1x3_bn', data=inception_resnet_v2_c3_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c3_1x3_scale = inception_resnet_v2_c3_1x3_bn
inception_resnet_v2_c3_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c3_1x3_relu', data=inception_resnet_v2_c3_1x3_scale , act_type='relu')
inception_resnet_v2_c3_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c3_3x1', data=inception_resnet_v2_c3_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c3_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c3_3x1_bn', data=inception_resnet_v2_c3_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c3_3x1_scale = inception_resnet_v2_c3_3x1_bn
inception_resnet_v2_c3_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c3_3x1_relu', data=inception_resnet_v2_c3_3x1_scale , act_type='relu')
inception_resnet_v2_c3_concat = mx.symbol.Concat(name='inception_resnet_v2_c3_concat', *[inception_resnet_v2_c3_1x1_relu,inception_resnet_v2_c3_3x1_relu] )
inception_resnet_v2_c3_up = mx.symbol.Convolution(name='inception_resnet_v2_c3_up', data=inception_resnet_v2_c3_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c3_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c3_residual_eltwise', *[1.0*inception_resnet_v2_c2_residual_eltwise_relu,0.20000000298*inception_resnet_v2_c3_up] )
inception_resnet_v2_c3_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_c3_residual_eltwise_relu', data=inception_resnet_v2_c3_residual_eltwise , act_type='relu')
inception_resnet_v2_c4_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c4_1x1', data=inception_resnet_v2_c3_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c4_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c4_1x1_bn', data=inception_resnet_v2_c4_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c4_1x1_scale = inception_resnet_v2_c4_1x1_bn
inception_resnet_v2_c4_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c4_1x1_relu', data=inception_resnet_v2_c4_1x1_scale , act_type='relu')
inception_resnet_v2_c4_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c4_1x3_reduce', data=inception_resnet_v2_c3_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c4_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c4_1x3_reduce_bn', data=inception_resnet_v2_c4_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c4_1x3_reduce_scale = inception_resnet_v2_c4_1x3_reduce_bn
inception_resnet_v2_c4_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c4_1x3_reduce_relu', data=inception_resnet_v2_c4_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c4_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c4_1x3', data=inception_resnet_v2_c4_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c4_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c4_1x3_bn', data=inception_resnet_v2_c4_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c4_1x3_scale = inception_resnet_v2_c4_1x3_bn
inception_resnet_v2_c4_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c4_1x3_relu', data=inception_resnet_v2_c4_1x3_scale , act_type='relu')
inception_resnet_v2_c4_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c4_3x1', data=inception_resnet_v2_c4_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c4_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c4_3x1_bn', data=inception_resnet_v2_c4_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c4_3x1_scale = inception_resnet_v2_c4_3x1_bn
inception_resnet_v2_c4_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c4_3x1_relu', data=inception_resnet_v2_c4_3x1_scale , act_type='relu')
inception_resnet_v2_c4_concat = mx.symbol.Concat(name='inception_resnet_v2_c4_concat', *[inception_resnet_v2_c4_1x1_relu,inception_resnet_v2_c4_3x1_relu] )
inception_resnet_v2_c4_up = mx.symbol.Convolution(name='inception_resnet_v2_c4_up', data=inception_resnet_v2_c4_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c4_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c4_residual_eltwise', *[1.0*inception_resnet_v2_c3_residual_eltwise_relu,0.20000000298*inception_resnet_v2_c4_up] )
inception_resnet_v2_c4_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_c4_residual_eltwise_relu', data=inception_resnet_v2_c4_residual_eltwise , act_type='relu')
inception_resnet_v2_c5_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c5_1x1', data=inception_resnet_v2_c4_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c5_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c5_1x1_bn', data=inception_resnet_v2_c5_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c5_1x1_scale = inception_resnet_v2_c5_1x1_bn
inception_resnet_v2_c5_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c5_1x1_relu', data=inception_resnet_v2_c5_1x1_scale , act_type='relu')
inception_resnet_v2_c5_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c5_1x3_reduce', data=inception_resnet_v2_c4_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c5_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c5_1x3_reduce_bn', data=inception_resnet_v2_c5_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c5_1x3_reduce_scale = inception_resnet_v2_c5_1x3_reduce_bn
inception_resnet_v2_c5_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c5_1x3_reduce_relu', data=inception_resnet_v2_c5_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c5_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c5_1x3', data=inception_resnet_v2_c5_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c5_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c5_1x3_bn', data=inception_resnet_v2_c5_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c5_1x3_scale = inception_resnet_v2_c5_1x3_bn
inception_resnet_v2_c5_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c5_1x3_relu', data=inception_resnet_v2_c5_1x3_scale , act_type='relu')
inception_resnet_v2_c5_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c5_3x1', data=inception_resnet_v2_c5_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c5_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c5_3x1_bn', data=inception_resnet_v2_c5_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c5_3x1_scale = inception_resnet_v2_c5_3x1_bn
inception_resnet_v2_c5_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c5_3x1_relu', data=inception_resnet_v2_c5_3x1_scale , act_type='relu')
inception_resnet_v2_c5_concat = mx.symbol.Concat(name='inception_resnet_v2_c5_concat', *[inception_resnet_v2_c5_1x1_relu,inception_resnet_v2_c5_3x1_relu] )
inception_resnet_v2_c5_up = mx.symbol.Convolution(name='inception_resnet_v2_c5_up', data=inception_resnet_v2_c5_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c5_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c5_residual_eltwise', *[1.0*inception_resnet_v2_c4_residual_eltwise_relu,0.20000000298*inception_resnet_v2_c5_up] )
inception_resnet_v2_c5_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_c5_residual_eltwise_relu', data=inception_resnet_v2_c5_residual_eltwise , act_type='relu')
inception_resnet_v2_c6_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c6_1x1', data=inception_resnet_v2_c5_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c6_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c6_1x1_bn', data=inception_resnet_v2_c6_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c6_1x1_scale = inception_resnet_v2_c6_1x1_bn
inception_resnet_v2_c6_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c6_1x1_relu', data=inception_resnet_v2_c6_1x1_scale , act_type='relu')
inception_resnet_v2_c6_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c6_1x3_reduce', data=inception_resnet_v2_c5_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c6_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c6_1x3_reduce_bn', data=inception_resnet_v2_c6_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c6_1x3_reduce_scale = inception_resnet_v2_c6_1x3_reduce_bn
inception_resnet_v2_c6_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c6_1x3_reduce_relu', data=inception_resnet_v2_c6_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c6_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c6_1x3', data=inception_resnet_v2_c6_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c6_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c6_1x3_bn', data=inception_resnet_v2_c6_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c6_1x3_scale = inception_resnet_v2_c6_1x3_bn
inception_resnet_v2_c6_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c6_1x3_relu', data=inception_resnet_v2_c6_1x3_scale , act_type='relu')
inception_resnet_v2_c6_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c6_3x1', data=inception_resnet_v2_c6_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c6_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c6_3x1_bn', data=inception_resnet_v2_c6_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c6_3x1_scale = inception_resnet_v2_c6_3x1_bn
inception_resnet_v2_c6_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c6_3x1_relu', data=inception_resnet_v2_c6_3x1_scale , act_type='relu')
inception_resnet_v2_c6_concat = mx.symbol.Concat(name='inception_resnet_v2_c6_concat', *[inception_resnet_v2_c6_1x1_relu,inception_resnet_v2_c6_3x1_relu] )
inception_resnet_v2_c6_up = mx.symbol.Convolution(name='inception_resnet_v2_c6_up', data=inception_resnet_v2_c6_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c6_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c6_residual_eltwise', *[1.0*inception_resnet_v2_c5_residual_eltwise_relu,0.20000000298*inception_resnet_v2_c6_up] )
inception_resnet_v2_c6_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_c6_residual_eltwise_relu', data=inception_resnet_v2_c6_residual_eltwise , act_type='relu')
inception_resnet_v2_c7_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c7_1x1', data=inception_resnet_v2_c6_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c7_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c7_1x1_bn', data=inception_resnet_v2_c7_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c7_1x1_scale = inception_resnet_v2_c7_1x1_bn
inception_resnet_v2_c7_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c7_1x1_relu', data=inception_resnet_v2_c7_1x1_scale , act_type='relu')
inception_resnet_v2_c7_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c7_1x3_reduce', data=inception_resnet_v2_c6_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c7_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c7_1x3_reduce_bn', data=inception_resnet_v2_c7_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c7_1x3_reduce_scale = inception_resnet_v2_c7_1x3_reduce_bn
inception_resnet_v2_c7_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c7_1x3_reduce_relu', data=inception_resnet_v2_c7_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c7_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c7_1x3', data=inception_resnet_v2_c7_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c7_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c7_1x3_bn', data=inception_resnet_v2_c7_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c7_1x3_scale = inception_resnet_v2_c7_1x3_bn
inception_resnet_v2_c7_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c7_1x3_relu', data=inception_resnet_v2_c7_1x3_scale , act_type='relu')
inception_resnet_v2_c7_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c7_3x1', data=inception_resnet_v2_c7_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c7_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c7_3x1_bn', data=inception_resnet_v2_c7_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c7_3x1_scale = inception_resnet_v2_c7_3x1_bn
inception_resnet_v2_c7_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c7_3x1_relu', data=inception_resnet_v2_c7_3x1_scale , act_type='relu')
inception_resnet_v2_c7_concat = mx.symbol.Concat(name='inception_resnet_v2_c7_concat', *[inception_resnet_v2_c7_1x1_relu,inception_resnet_v2_c7_3x1_relu] )
inception_resnet_v2_c7_up = mx.symbol.Convolution(name='inception_resnet_v2_c7_up', data=inception_resnet_v2_c7_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c7_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c7_residual_eltwise', *[1.0*inception_resnet_v2_c6_residual_eltwise_relu,0.20000000298*inception_resnet_v2_c7_up] )
inception_resnet_v2_c7_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_c7_residual_eltwise_relu', data=inception_resnet_v2_c7_residual_eltwise , act_type='relu')
inception_resnet_v2_c8_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c8_1x1', data=inception_resnet_v2_c7_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c8_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c8_1x1_bn', data=inception_resnet_v2_c8_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c8_1x1_scale = inception_resnet_v2_c8_1x1_bn
inception_resnet_v2_c8_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c8_1x1_relu', data=inception_resnet_v2_c8_1x1_scale , act_type='relu')
inception_resnet_v2_c8_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c8_1x3_reduce', data=inception_resnet_v2_c7_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c8_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c8_1x3_reduce_bn', data=inception_resnet_v2_c8_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c8_1x3_reduce_scale = inception_resnet_v2_c8_1x3_reduce_bn
inception_resnet_v2_c8_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c8_1x3_reduce_relu', data=inception_resnet_v2_c8_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c8_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c8_1x3', data=inception_resnet_v2_c8_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c8_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c8_1x3_bn', data=inception_resnet_v2_c8_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c8_1x3_scale = inception_resnet_v2_c8_1x3_bn
inception_resnet_v2_c8_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c8_1x3_relu', data=inception_resnet_v2_c8_1x3_scale , act_type='relu')
inception_resnet_v2_c8_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c8_3x1', data=inception_resnet_v2_c8_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c8_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c8_3x1_bn', data=inception_resnet_v2_c8_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c8_3x1_scale = inception_resnet_v2_c8_3x1_bn
inception_resnet_v2_c8_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c8_3x1_relu', data=inception_resnet_v2_c8_3x1_scale , act_type='relu')
inception_resnet_v2_c8_concat = mx.symbol.Concat(name='inception_resnet_v2_c8_concat', *[inception_resnet_v2_c8_1x1_relu,inception_resnet_v2_c8_3x1_relu] )
inception_resnet_v2_c8_up = mx.symbol.Convolution(name='inception_resnet_v2_c8_up', data=inception_resnet_v2_c8_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c8_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c8_residual_eltwise', *[1.0*inception_resnet_v2_c7_residual_eltwise_relu,0.20000000298*inception_resnet_v2_c8_up] )
inception_resnet_v2_c8_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_c8_residual_eltwise_relu', data=inception_resnet_v2_c8_residual_eltwise , act_type='relu')
inception_resnet_v2_c9_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c9_1x1', data=inception_resnet_v2_c8_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c9_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c9_1x1_bn', data=inception_resnet_v2_c9_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c9_1x1_scale = inception_resnet_v2_c9_1x1_bn
inception_resnet_v2_c9_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c9_1x1_relu', data=inception_resnet_v2_c9_1x1_scale , act_type='relu')
inception_resnet_v2_c9_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c9_1x3_reduce', data=inception_resnet_v2_c8_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c9_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c9_1x3_reduce_bn', data=inception_resnet_v2_c9_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c9_1x3_reduce_scale = inception_resnet_v2_c9_1x3_reduce_bn
inception_resnet_v2_c9_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c9_1x3_reduce_relu', data=inception_resnet_v2_c9_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c9_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c9_1x3', data=inception_resnet_v2_c9_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c9_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c9_1x3_bn', data=inception_resnet_v2_c9_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c9_1x3_scale = inception_resnet_v2_c9_1x3_bn
inception_resnet_v2_c9_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c9_1x3_relu', data=inception_resnet_v2_c9_1x3_scale , act_type='relu')
inception_resnet_v2_c9_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c9_3x1', data=inception_resnet_v2_c9_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c9_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c9_3x1_bn', data=inception_resnet_v2_c9_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c9_3x1_scale = inception_resnet_v2_c9_3x1_bn
inception_resnet_v2_c9_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c9_3x1_relu', data=inception_resnet_v2_c9_3x1_scale , act_type='relu')
inception_resnet_v2_c9_concat = mx.symbol.Concat(name='inception_resnet_v2_c9_concat', *[inception_resnet_v2_c9_1x1_relu,inception_resnet_v2_c9_3x1_relu] )
inception_resnet_v2_c9_up = mx.symbol.Convolution(name='inception_resnet_v2_c9_up', data=inception_resnet_v2_c9_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c9_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c9_residual_eltwise', *[1.0*inception_resnet_v2_c8_residual_eltwise_relu,0.20000000298*inception_resnet_v2_c9_up] )
inception_resnet_v2_c9_residual_eltwise_relu = mx.symbol.Activation(name='inception_resnet_v2_c9_residual_eltwise_relu', data=inception_resnet_v2_c9_residual_eltwise , act_type='relu')
inception_resnet_v2_c10_1x1 = mx.symbol.Convolution(name='inception_resnet_v2_c10_1x1', data=inception_resnet_v2_c9_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c10_1x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c10_1x1_bn', data=inception_resnet_v2_c10_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c10_1x1_scale = inception_resnet_v2_c10_1x1_bn
inception_resnet_v2_c10_1x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c10_1x1_relu', data=inception_resnet_v2_c10_1x1_scale , act_type='relu')
inception_resnet_v2_c10_1x3_reduce = mx.symbol.Convolution(name='inception_resnet_v2_c10_1x3_reduce', data=inception_resnet_v2_c9_residual_eltwise_relu , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c10_1x3_reduce_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c10_1x3_reduce_bn', data=inception_resnet_v2_c10_1x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c10_1x3_reduce_scale = inception_resnet_v2_c10_1x3_reduce_bn
inception_resnet_v2_c10_1x3_reduce_relu = mx.symbol.Activation(name='inception_resnet_v2_c10_1x3_reduce_relu', data=inception_resnet_v2_c10_1x3_reduce_scale , act_type='relu')
inception_resnet_v2_c10_1x3 = mx.symbol.Convolution(name='inception_resnet_v2_c10_1x3', data=inception_resnet_v2_c10_1x3_reduce_relu , num_filter=224, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_resnet_v2_c10_1x3_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c10_1x3_bn', data=inception_resnet_v2_c10_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c10_1x3_scale = inception_resnet_v2_c10_1x3_bn
inception_resnet_v2_c10_1x3_relu = mx.symbol.Activation(name='inception_resnet_v2_c10_1x3_relu', data=inception_resnet_v2_c10_1x3_scale , act_type='relu')
inception_resnet_v2_c10_3x1 = mx.symbol.Convolution(name='inception_resnet_v2_c10_3x1', data=inception_resnet_v2_c10_1x3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_resnet_v2_c10_3x1_bn = mx.symbol.BatchNorm(name='inception_resnet_v2_c10_3x1_bn', data=inception_resnet_v2_c10_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_resnet_v2_c10_3x1_scale = inception_resnet_v2_c10_3x1_bn
inception_resnet_v2_c10_3x1_relu = mx.symbol.Activation(name='inception_resnet_v2_c10_3x1_relu', data=inception_resnet_v2_c10_3x1_scale , act_type='relu')
inception_resnet_v2_c10_concat = mx.symbol.Concat(name='inception_resnet_v2_c10_concat', *[inception_resnet_v2_c10_1x1_relu,inception_resnet_v2_c10_3x1_relu] )
inception_resnet_v2_c10_up = mx.symbol.Convolution(name='inception_resnet_v2_c10_up', data=inception_resnet_v2_c10_concat , num_filter=2080, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
inception_resnet_v2_c10_residual_eltwise = mx.symbol.broadcast_add(name='inception_resnet_v2_c10_residual_eltwise', *[1.0*inception_resnet_v2_c9_residual_eltwise_relu,1.0*inception_resnet_v2_c10_up] )
conv6_1x1 = mx.symbol.Convolution(name='conv6_1x1', data=inception_resnet_v2_c10_residual_eltwise , num_filter=1536, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
conv6_1x1_bn = mx.symbol.BatchNorm(name='conv6_1x1_bn', data=conv6_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv6_1x1_scale = conv6_1x1_bn
conv6_1x1_relu = mx.symbol.Activation(name='conv6_1x1_relu', data=conv6_1x1_scale , act_type='relu')
pool_8x8_s1 = mx.symbol.Pooling(name='pool_8x8_s1', data=conv6_1x1_relu , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
pool_8x8_s1_drop = mx.symbol.Dropout(name='pool_8x8_s1_drop', data=pool_8x8_s1 , p=0.200000)
flatten_0=mx.symbol.Flatten(name='flatten_0', data=pool_8x8_s1_drop)
classifier = mx.symbol.FullyConnected(name='classifier', data=flatten_0 , num_hidden=1000, no_bias=False)
prob = mx.symbol.SoftmaxOutput(name='prob', data=classifier )
