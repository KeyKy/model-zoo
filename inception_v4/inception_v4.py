import mxnet as mx
data = mx.symbol.Variable(name='data')
conv1_3x3_s2 = mx.symbol.Convolution(name='conv1_3x3_s2', data=data , num_filter=32, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
conv1_3x3_s2_bn = mx.symbol.BatchNorm(name='conv1_3x3_s2_bn', data=conv1_3x3_s2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv1_3x3_s2_scale = conv1_3x3_s2_bn
conv1_3x3_s2_relu = mx.symbol.Activation(name='conv1_3x3_s2_relu', data=conv1_3x3_s2_scale , act_type='relu')
conv2_3x3_s1 = mx.symbol.Convolution(name='conv2_3x3_s1', data=conv1_3x3_s2_relu , num_filter=32, pad=(0, 0), kernel=(3,3), stride=(1,1), no_bias=True)
conv2_3x3_s1_bn = mx.symbol.BatchNorm(name='conv2_3x3_s1_bn', data=conv2_3x3_s1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv2_3x3_s1_scale = conv2_3x3_s1_bn
conv2_3x3_s1_relu = mx.symbol.Activation(name='conv2_3x3_s1_relu', data=conv2_3x3_s1_scale , act_type='relu')
conv3_3x3_s1 = mx.symbol.Convolution(name='conv3_3x3_s1', data=conv2_3x3_s1_relu , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
conv3_3x3_s1_bn = mx.symbol.BatchNorm(name='conv3_3x3_s1_bn', data=conv3_3x3_s1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
conv3_3x3_s1_scale = conv3_3x3_s1_bn
conv3_3x3_s1_relu = mx.symbol.Activation(name='conv3_3x3_s1_relu', data=conv3_3x3_s1_scale , act_type='relu')
inception_stem1_3x3_s2 = mx.symbol.Convolution(name='inception_stem1_3x3_s2', data=conv3_3x3_s1_relu , num_filter=96, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
inception_stem1_3x3_s2_bn = mx.symbol.BatchNorm(name='inception_stem1_3x3_s2_bn', data=inception_stem1_3x3_s2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_stem1_3x3_s2_scale = inception_stem1_3x3_s2_bn
inception_stem1_3x3_s2_relu = mx.symbol.Activation(name='inception_stem1_3x3_s2_relu', data=inception_stem1_3x3_s2_scale , act_type='relu')
inception_stem1_pool = mx.symbol.Pooling(name='inception_stem1_pool', data=conv3_3x3_s1_relu , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
inception_stem1 = mx.symbol.Concat(name='inception_stem1', *[inception_stem1_pool,inception_stem1_3x3_s2_relu] )
inception_stem2_3x3_reduce = mx.symbol.Convolution(name='inception_stem2_3x3_reduce', data=inception_stem1 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_stem2_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_stem2_3x3_reduce_bn', data=inception_stem2_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_stem2_3x3_reduce_scale = inception_stem2_3x3_reduce_bn
inception_stem2_3x3_reduce_relu = mx.symbol.Activation(name='inception_stem2_3x3_reduce_relu', data=inception_stem2_3x3_reduce_scale , act_type='relu')
inception_stem2_3x3 = mx.symbol.Convolution(name='inception_stem2_3x3', data=inception_stem2_3x3_reduce_relu , num_filter=96, pad=(0, 0), kernel=(3,3), stride=(1,1), no_bias=True)
inception_stem2_3x3_bn = mx.symbol.BatchNorm(name='inception_stem2_3x3_bn', data=inception_stem2_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_stem2_3x3_scale = inception_stem2_3x3_bn
inception_stem2_3x3_relu = mx.symbol.Activation(name='inception_stem2_3x3_relu', data=inception_stem2_3x3_scale , act_type='relu')
inception_stem2_1x7_reduce = mx.symbol.Convolution(name='inception_stem2_1x7_reduce', data=inception_stem1 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_stem2_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_stem2_1x7_reduce_bn', data=inception_stem2_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_stem2_1x7_reduce_scale = inception_stem2_1x7_reduce_bn
inception_stem2_1x7_reduce_relu = mx.symbol.Activation(name='inception_stem2_1x7_reduce_relu', data=inception_stem2_1x7_reduce_scale , act_type='relu')
inception_stem2_1x7 = mx.symbol.Convolution(name='inception_stem2_1x7', data=inception_stem2_1x7_reduce_relu , num_filter=64, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_stem2_1x7_bn = mx.symbol.BatchNorm(name='inception_stem2_1x7_bn', data=inception_stem2_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_stem2_1x7_scale = inception_stem2_1x7_bn
inception_stem2_1x7_relu = mx.symbol.Activation(name='inception_stem2_1x7_relu', data=inception_stem2_1x7_scale , act_type='relu')
inception_stem2_7x1 = mx.symbol.Convolution(name='inception_stem2_7x1', data=inception_stem2_1x7_relu , num_filter=64, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_stem2_7x1_bn = mx.symbol.BatchNorm(name='inception_stem2_7x1_bn', data=inception_stem2_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_stem2_7x1_scale = inception_stem2_7x1_bn
inception_stem2_7x1_relu = mx.symbol.Activation(name='inception_stem2_7x1_relu', data=inception_stem2_7x1_scale , act_type='relu')
inception_stem2_3x3_2 = mx.symbol.Convolution(name='inception_stem2_3x3_2', data=inception_stem2_7x1_relu , num_filter=96, pad=(0, 0), kernel=(3,3), stride=(1,1), no_bias=True)
inception_stem2_3x3_2_bn = mx.symbol.BatchNorm(name='inception_stem2_3x3_2_bn', data=inception_stem2_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_stem2_3x3_2_scale = inception_stem2_3x3_2_bn
inception_stem2_3x3_2_relu = mx.symbol.Activation(name='inception_stem2_3x3_2_relu', data=inception_stem2_3x3_2_scale , act_type='relu')
inception_stem2 = mx.symbol.Concat(name='inception_stem2', *[inception_stem2_3x3_relu,inception_stem2_3x3_2_relu] )
inception_stem3_3x3_s2 = mx.symbol.Convolution(name='inception_stem3_3x3_s2', data=inception_stem2 , num_filter=192, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
inception_stem3_3x3_s2_bn = mx.symbol.BatchNorm(name='inception_stem3_3x3_s2_bn', data=inception_stem3_3x3_s2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_stem3_3x3_s2_scale = inception_stem3_3x3_s2_bn
inception_stem3_3x3_s2_relu = mx.symbol.Activation(name='inception_stem3_3x3_s2_relu', data=inception_stem3_3x3_s2_scale , act_type='relu')
inception_stem3_pool = mx.symbol.Pooling(name='inception_stem3_pool', data=inception_stem2 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
inception_stem3 = mx.symbol.Concat(name='inception_stem3', *[inception_stem3_3x3_s2_relu,inception_stem3_pool] )
inception_a1_1x1_2 = mx.symbol.Convolution(name='inception_a1_1x1_2', data=inception_stem3 , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a1_1x1_2_bn = mx.symbol.BatchNorm(name='inception_a1_1x1_2_bn', data=inception_a1_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a1_1x1_2_scale = inception_a1_1x1_2_bn
inception_a1_1x1_2_relu = mx.symbol.Activation(name='inception_a1_1x1_2_relu', data=inception_a1_1x1_2_scale , act_type='relu')
inception_a1_3x3_reduce = mx.symbol.Convolution(name='inception_a1_3x3_reduce', data=inception_stem3 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a1_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_a1_3x3_reduce_bn', data=inception_a1_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a1_3x3_reduce_scale = inception_a1_3x3_reduce_bn
inception_a1_3x3_reduce_relu = mx.symbol.Activation(name='inception_a1_3x3_reduce_relu', data=inception_a1_3x3_reduce_scale , act_type='relu')
inception_a1_3x3 = mx.symbol.Convolution(name='inception_a1_3x3', data=inception_a1_3x3_reduce_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a1_3x3_bn = mx.symbol.BatchNorm(name='inception_a1_3x3_bn', data=inception_a1_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a1_3x3_scale = inception_a1_3x3_bn
inception_a1_3x3_relu = mx.symbol.Activation(name='inception_a1_3x3_relu', data=inception_a1_3x3_scale , act_type='relu')
inception_a1_3x3_2_reduce = mx.symbol.Convolution(name='inception_a1_3x3_2_reduce', data=inception_stem3 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a1_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_a1_3x3_2_reduce_bn', data=inception_a1_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a1_3x3_2_reduce_scale = inception_a1_3x3_2_reduce_bn
inception_a1_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_a1_3x3_2_reduce_relu', data=inception_a1_3x3_2_reduce_scale , act_type='relu')
inception_a1_3x3_2 = mx.symbol.Convolution(name='inception_a1_3x3_2', data=inception_a1_3x3_2_reduce_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a1_3x3_2_bn = mx.symbol.BatchNorm(name='inception_a1_3x3_2_bn', data=inception_a1_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a1_3x3_2_scale = inception_a1_3x3_2_bn
inception_a1_3x3_2_relu = mx.symbol.Activation(name='inception_a1_3x3_2_relu', data=inception_a1_3x3_2_scale , act_type='relu')
inception_a1_3x3_3 = mx.symbol.Convolution(name='inception_a1_3x3_3', data=inception_a1_3x3_2_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a1_3x3_3_bn = mx.symbol.BatchNorm(name='inception_a1_3x3_3_bn', data=inception_a1_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a1_3x3_3_scale = inception_a1_3x3_3_bn
inception_a1_3x3_3_relu = mx.symbol.Activation(name='inception_a1_3x3_3_relu', data=inception_a1_3x3_3_scale , act_type='relu')
inception_a1_pool_ave = mx.symbol.Pooling(name='inception_a1_pool_ave', data=inception_stem3 , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_a1_1x1 = mx.symbol.Convolution(name='inception_a1_1x1', data=inception_a1_pool_ave , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a1_1x1_bn = mx.symbol.BatchNorm(name='inception_a1_1x1_bn', data=inception_a1_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a1_1x1_scale = inception_a1_1x1_bn
inception_a1_1x1_relu = mx.symbol.Activation(name='inception_a1_1x1_relu', data=inception_a1_1x1_scale , act_type='relu')
inception_a1_concat = mx.symbol.Concat(name='inception_a1_concat', *[inception_a1_1x1_2_relu,inception_a1_3x3_relu,inception_a1_3x3_3_relu,inception_a1_1x1_relu] )
inception_a2_1x1_2 = mx.symbol.Convolution(name='inception_a2_1x1_2', data=inception_a1_concat , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a2_1x1_2_bn = mx.symbol.BatchNorm(name='inception_a2_1x1_2_bn', data=inception_a2_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a2_1x1_2_scale = inception_a2_1x1_2_bn
inception_a2_1x1_2_relu = mx.symbol.Activation(name='inception_a2_1x1_2_relu', data=inception_a2_1x1_2_scale , act_type='relu')
inception_a2_3x3_reduce = mx.symbol.Convolution(name='inception_a2_3x3_reduce', data=inception_a1_concat , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a2_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_a2_3x3_reduce_bn', data=inception_a2_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a2_3x3_reduce_scale = inception_a2_3x3_reduce_bn
inception_a2_3x3_reduce_relu = mx.symbol.Activation(name='inception_a2_3x3_reduce_relu', data=inception_a2_3x3_reduce_scale , act_type='relu')
inception_a2_3x3 = mx.symbol.Convolution(name='inception_a2_3x3', data=inception_a2_3x3_reduce_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a2_3x3_bn = mx.symbol.BatchNorm(name='inception_a2_3x3_bn', data=inception_a2_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a2_3x3_scale = inception_a2_3x3_bn
inception_a2_3x3_relu = mx.symbol.Activation(name='inception_a2_3x3_relu', data=inception_a2_3x3_scale , act_type='relu')
inception_a2_3x3_2_reduce = mx.symbol.Convolution(name='inception_a2_3x3_2_reduce', data=inception_a1_concat , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a2_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_a2_3x3_2_reduce_bn', data=inception_a2_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a2_3x3_2_reduce_scale = inception_a2_3x3_2_reduce_bn
inception_a2_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_a2_3x3_2_reduce_relu', data=inception_a2_3x3_2_reduce_scale , act_type='relu')
inception_a2_3x3_2 = mx.symbol.Convolution(name='inception_a2_3x3_2', data=inception_a2_3x3_2_reduce_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a2_3x3_2_bn = mx.symbol.BatchNorm(name='inception_a2_3x3_2_bn', data=inception_a2_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a2_3x3_2_scale = inception_a2_3x3_2_bn
inception_a2_3x3_2_relu = mx.symbol.Activation(name='inception_a2_3x3_2_relu', data=inception_a2_3x3_2_scale , act_type='relu')
inception_a2_3x3_3 = mx.symbol.Convolution(name='inception_a2_3x3_3', data=inception_a2_3x3_2_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a2_3x3_3_bn = mx.symbol.BatchNorm(name='inception_a2_3x3_3_bn', data=inception_a2_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a2_3x3_3_scale = inception_a2_3x3_3_bn
inception_a2_3x3_3_relu = mx.symbol.Activation(name='inception_a2_3x3_3_relu', data=inception_a2_3x3_3_scale , act_type='relu')
inception_a2_pool_ave = mx.symbol.Pooling(name='inception_a2_pool_ave', data=inception_a1_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_a2_1x1 = mx.symbol.Convolution(name='inception_a2_1x1', data=inception_a2_pool_ave , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a2_1x1_bn = mx.symbol.BatchNorm(name='inception_a2_1x1_bn', data=inception_a2_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a2_1x1_scale = inception_a2_1x1_bn
inception_a2_1x1_relu = mx.symbol.Activation(name='inception_a2_1x1_relu', data=inception_a2_1x1_scale , act_type='relu')
inception_a2_concat = mx.symbol.Concat(name='inception_a2_concat', *[inception_a2_1x1_2_relu,inception_a2_3x3_relu,inception_a2_3x3_3_relu,inception_a2_1x1_relu] )
inception_a3_1x1_2 = mx.symbol.Convolution(name='inception_a3_1x1_2', data=inception_a2_concat , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a3_1x1_2_bn = mx.symbol.BatchNorm(name='inception_a3_1x1_2_bn', data=inception_a3_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a3_1x1_2_scale = inception_a3_1x1_2_bn
inception_a3_1x1_2_relu = mx.symbol.Activation(name='inception_a3_1x1_2_relu', data=inception_a3_1x1_2_scale , act_type='relu')
inception_a3_3x3_reduce = mx.symbol.Convolution(name='inception_a3_3x3_reduce', data=inception_a2_concat , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a3_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_a3_3x3_reduce_bn', data=inception_a3_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a3_3x3_reduce_scale = inception_a3_3x3_reduce_bn
inception_a3_3x3_reduce_relu = mx.symbol.Activation(name='inception_a3_3x3_reduce_relu', data=inception_a3_3x3_reduce_scale , act_type='relu')
inception_a3_3x3 = mx.symbol.Convolution(name='inception_a3_3x3', data=inception_a3_3x3_reduce_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a3_3x3_bn = mx.symbol.BatchNorm(name='inception_a3_3x3_bn', data=inception_a3_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a3_3x3_scale = inception_a3_3x3_bn
inception_a3_3x3_relu = mx.symbol.Activation(name='inception_a3_3x3_relu', data=inception_a3_3x3_scale , act_type='relu')
inception_a3_3x3_2_reduce = mx.symbol.Convolution(name='inception_a3_3x3_2_reduce', data=inception_a2_concat , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a3_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_a3_3x3_2_reduce_bn', data=inception_a3_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a3_3x3_2_reduce_scale = inception_a3_3x3_2_reduce_bn
inception_a3_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_a3_3x3_2_reduce_relu', data=inception_a3_3x3_2_reduce_scale , act_type='relu')
inception_a3_3x3_2 = mx.symbol.Convolution(name='inception_a3_3x3_2', data=inception_a3_3x3_2_reduce_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a3_3x3_2_bn = mx.symbol.BatchNorm(name='inception_a3_3x3_2_bn', data=inception_a3_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a3_3x3_2_scale = inception_a3_3x3_2_bn
inception_a3_3x3_2_relu = mx.symbol.Activation(name='inception_a3_3x3_2_relu', data=inception_a3_3x3_2_scale , act_type='relu')
inception_a3_3x3_3 = mx.symbol.Convolution(name='inception_a3_3x3_3', data=inception_a3_3x3_2_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a3_3x3_3_bn = mx.symbol.BatchNorm(name='inception_a3_3x3_3_bn', data=inception_a3_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a3_3x3_3_scale = inception_a3_3x3_3_bn
inception_a3_3x3_3_relu = mx.symbol.Activation(name='inception_a3_3x3_3_relu', data=inception_a3_3x3_3_scale , act_type='relu')
inception_a3_pool_ave = mx.symbol.Pooling(name='inception_a3_pool_ave', data=inception_a2_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_a3_1x1 = mx.symbol.Convolution(name='inception_a3_1x1', data=inception_a3_pool_ave , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a3_1x1_bn = mx.symbol.BatchNorm(name='inception_a3_1x1_bn', data=inception_a3_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a3_1x1_scale = inception_a3_1x1_bn
inception_a3_1x1_relu = mx.symbol.Activation(name='inception_a3_1x1_relu', data=inception_a3_1x1_scale , act_type='relu')
inception_a3_concat = mx.symbol.Concat(name='inception_a3_concat', *[inception_a3_1x1_2_relu,inception_a3_3x3_relu,inception_a3_3x3_3_relu,inception_a3_1x1_relu] )
inception_a4_1x1_2 = mx.symbol.Convolution(name='inception_a4_1x1_2', data=inception_a3_concat , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a4_1x1_2_bn = mx.symbol.BatchNorm(name='inception_a4_1x1_2_bn', data=inception_a4_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a4_1x1_2_scale = inception_a4_1x1_2_bn
inception_a4_1x1_2_relu = mx.symbol.Activation(name='inception_a4_1x1_2_relu', data=inception_a4_1x1_2_scale , act_type='relu')
inception_a4_3x3_reduce = mx.symbol.Convolution(name='inception_a4_3x3_reduce', data=inception_a3_concat , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a4_3x3_reduce_bn = mx.symbol.BatchNorm(name='inception_a4_3x3_reduce_bn', data=inception_a4_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a4_3x3_reduce_scale = inception_a4_3x3_reduce_bn
inception_a4_3x3_reduce_relu = mx.symbol.Activation(name='inception_a4_3x3_reduce_relu', data=inception_a4_3x3_reduce_scale , act_type='relu')
inception_a4_3x3 = mx.symbol.Convolution(name='inception_a4_3x3', data=inception_a4_3x3_reduce_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a4_3x3_bn = mx.symbol.BatchNorm(name='inception_a4_3x3_bn', data=inception_a4_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a4_3x3_scale = inception_a4_3x3_bn
inception_a4_3x3_relu = mx.symbol.Activation(name='inception_a4_3x3_relu', data=inception_a4_3x3_scale , act_type='relu')
inception_a4_3x3_2_reduce = mx.symbol.Convolution(name='inception_a4_3x3_2_reduce', data=inception_a3_concat , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a4_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='inception_a4_3x3_2_reduce_bn', data=inception_a4_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a4_3x3_2_reduce_scale = inception_a4_3x3_2_reduce_bn
inception_a4_3x3_2_reduce_relu = mx.symbol.Activation(name='inception_a4_3x3_2_reduce_relu', data=inception_a4_3x3_2_reduce_scale , act_type='relu')
inception_a4_3x3_2 = mx.symbol.Convolution(name='inception_a4_3x3_2', data=inception_a4_3x3_2_reduce_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a4_3x3_2_bn = mx.symbol.BatchNorm(name='inception_a4_3x3_2_bn', data=inception_a4_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a4_3x3_2_scale = inception_a4_3x3_2_bn
inception_a4_3x3_2_relu = mx.symbol.Activation(name='inception_a4_3x3_2_relu', data=inception_a4_3x3_2_scale , act_type='relu')
inception_a4_3x3_3 = mx.symbol.Convolution(name='inception_a4_3x3_3', data=inception_a4_3x3_2_relu , num_filter=96, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
inception_a4_3x3_3_bn = mx.symbol.BatchNorm(name='inception_a4_3x3_3_bn', data=inception_a4_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a4_3x3_3_scale = inception_a4_3x3_3_bn
inception_a4_3x3_3_relu = mx.symbol.Activation(name='inception_a4_3x3_3_relu', data=inception_a4_3x3_3_scale , act_type='relu')
inception_a4_pool_ave = mx.symbol.Pooling(name='inception_a4_pool_ave', data=inception_a3_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_a4_1x1 = mx.symbol.Convolution(name='inception_a4_1x1', data=inception_a4_pool_ave , num_filter=96, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_a4_1x1_bn = mx.symbol.BatchNorm(name='inception_a4_1x1_bn', data=inception_a4_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_a4_1x1_scale = inception_a4_1x1_bn
inception_a4_1x1_relu = mx.symbol.Activation(name='inception_a4_1x1_relu', data=inception_a4_1x1_scale , act_type='relu')
inception_a4_concat = mx.symbol.Concat(name='inception_a4_concat', *[inception_a4_1x1_2_relu,inception_a4_3x3_relu,inception_a4_3x3_3_relu,inception_a4_1x1_relu] )
reduction_a_3x3 = mx.symbol.Convolution(name='reduction_a_3x3', data=inception_a4_concat , num_filter=384, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
reduction_a_3x3_bn = mx.symbol.BatchNorm(name='reduction_a_3x3_bn', data=reduction_a_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_a_3x3_scale = reduction_a_3x3_bn
reduction_a_3x3_relu = mx.symbol.Activation(name='reduction_a_3x3_relu', data=reduction_a_3x3_scale , act_type='relu')
reduction_a_3x3_2_reduce = mx.symbol.Convolution(name='reduction_a_3x3_2_reduce', data=inception_a4_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduction_a_3x3_2_reduce_bn = mx.symbol.BatchNorm(name='reduction_a_3x3_2_reduce_bn', data=reduction_a_3x3_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_a_3x3_2_reduce_scale = reduction_a_3x3_2_reduce_bn
reduction_a_3x3_2_reduce_relu = mx.symbol.Activation(name='reduction_a_3x3_2_reduce_relu', data=reduction_a_3x3_2_reduce_scale , act_type='relu')
reduction_a_3x3_2 = mx.symbol.Convolution(name='reduction_a_3x3_2', data=reduction_a_3x3_2_reduce_relu , num_filter=224, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
reduction_a_3x3_2_bn = mx.symbol.BatchNorm(name='reduction_a_3x3_2_bn', data=reduction_a_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_a_3x3_2_scale = reduction_a_3x3_2_bn
reduction_a_3x3_2_relu = mx.symbol.Activation(name='reduction_a_3x3_2_relu', data=reduction_a_3x3_2_scale , act_type='relu')
reduction_a_3x3_3 = mx.symbol.Convolution(name='reduction_a_3x3_3', data=reduction_a_3x3_2_relu , num_filter=256, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
reduction_a_3x3_3_bn = mx.symbol.BatchNorm(name='reduction_a_3x3_3_bn', data=reduction_a_3x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_a_3x3_3_scale = reduction_a_3x3_3_bn
reduction_a_3x3_3_relu = mx.symbol.Activation(name='reduction_a_3x3_3_relu', data=reduction_a_3x3_3_scale , act_type='relu')
reduction_a_pool = mx.symbol.Pooling(name='reduction_a_pool', data=inception_a4_concat , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
reduction_a_concat = mx.symbol.Concat(name='reduction_a_concat', *[reduction_a_3x3_relu,reduction_a_3x3_3_relu,reduction_a_pool] )
inception_b1_1x1_2 = mx.symbol.Convolution(name='inception_b1_1x1_2', data=reduction_a_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b1_1x1_2_bn = mx.symbol.BatchNorm(name='inception_b1_1x1_2_bn', data=inception_b1_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_1x1_2_scale = inception_b1_1x1_2_bn
inception_b1_1x1_2_relu = mx.symbol.Activation(name='inception_b1_1x1_2_relu', data=inception_b1_1x1_2_scale , act_type='relu')
inception_b1_1x7_reduce = mx.symbol.Convolution(name='inception_b1_1x7_reduce', data=reduction_a_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b1_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_b1_1x7_reduce_bn', data=inception_b1_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_1x7_reduce_scale = inception_b1_1x7_reduce_bn
inception_b1_1x7_reduce_relu = mx.symbol.Activation(name='inception_b1_1x7_reduce_relu', data=inception_b1_1x7_reduce_scale , act_type='relu')
inception_b1_1x7 = mx.symbol.Convolution(name='inception_b1_1x7', data=inception_b1_1x7_reduce_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b1_1x7_bn = mx.symbol.BatchNorm(name='inception_b1_1x7_bn', data=inception_b1_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_1x7_scale = inception_b1_1x7_bn
inception_b1_1x7_relu = mx.symbol.Activation(name='inception_b1_1x7_relu', data=inception_b1_1x7_scale , act_type='relu')
inception_b1_7x1 = mx.symbol.Convolution(name='inception_b1_7x1', data=inception_b1_1x7_relu , num_filter=256, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b1_7x1_bn = mx.symbol.BatchNorm(name='inception_b1_7x1_bn', data=inception_b1_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_7x1_scale = inception_b1_7x1_bn
inception_b1_7x1_relu = mx.symbol.Activation(name='inception_b1_7x1_relu', data=inception_b1_7x1_scale , act_type='relu')
inception_b1_7x1_2_reduce = mx.symbol.Convolution(name='inception_b1_7x1_2_reduce', data=reduction_a_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b1_7x1_2_reduce_bn = mx.symbol.BatchNorm(name='inception_b1_7x1_2_reduce_bn', data=inception_b1_7x1_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_7x1_2_reduce_scale = inception_b1_7x1_2_reduce_bn
inception_b1_7x1_2_reduce_relu = mx.symbol.Activation(name='inception_b1_7x1_2_reduce_relu', data=inception_b1_7x1_2_reduce_scale , act_type='relu')
inception_b1_7x1_2 = mx.symbol.Convolution(name='inception_b1_7x1_2', data=inception_b1_7x1_2_reduce_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b1_7x1_2_bn = mx.symbol.BatchNorm(name='inception_b1_7x1_2_bn', data=inception_b1_7x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_7x1_2_scale = inception_b1_7x1_2_bn
inception_b1_7x1_2_relu = mx.symbol.Activation(name='inception_b1_7x1_2_relu', data=inception_b1_7x1_2_scale , act_type='relu')
inception_b1_1x7_2 = mx.symbol.Convolution(name='inception_b1_1x7_2', data=inception_b1_7x1_2_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b1_1x7_2_bn = mx.symbol.BatchNorm(name='inception_b1_1x7_2_bn', data=inception_b1_1x7_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_1x7_2_scale = inception_b1_1x7_2_bn
inception_b1_1x7_2_relu = mx.symbol.Activation(name='inception_b1_1x7_2_relu', data=inception_b1_1x7_2_scale , act_type='relu')
inception_b1_7x1_3 = mx.symbol.Convolution(name='inception_b1_7x1_3', data=inception_b1_1x7_2_relu , num_filter=224, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b1_7x1_3_bn = mx.symbol.BatchNorm(name='inception_b1_7x1_3_bn', data=inception_b1_7x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_7x1_3_scale = inception_b1_7x1_3_bn
inception_b1_7x1_3_relu = mx.symbol.Activation(name='inception_b1_7x1_3_relu', data=inception_b1_7x1_3_scale , act_type='relu')
inception_b1_1x7_3 = mx.symbol.Convolution(name='inception_b1_1x7_3', data=inception_b1_7x1_3_relu , num_filter=256, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b1_1x7_3_bn = mx.symbol.BatchNorm(name='inception_b1_1x7_3_bn', data=inception_b1_1x7_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_1x7_3_scale = inception_b1_1x7_3_bn
inception_b1_1x7_3_relu = mx.symbol.Activation(name='inception_b1_1x7_3_relu', data=inception_b1_1x7_3_scale , act_type='relu')
inception_b1_pool_ave = mx.symbol.Pooling(name='inception_b1_pool_ave', data=reduction_a_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_b1_1x1 = mx.symbol.Convolution(name='inception_b1_1x1', data=inception_b1_pool_ave , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b1_1x1_bn = mx.symbol.BatchNorm(name='inception_b1_1x1_bn', data=inception_b1_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b1_1x1_scale = inception_b1_1x1_bn
inception_b1_1x1_relu = mx.symbol.Activation(name='inception_b1_1x1_relu', data=inception_b1_1x1_scale , act_type='relu')
inception_b1_concat = mx.symbol.Concat(name='inception_b1_concat', *[inception_b1_1x1_2_relu,inception_b1_7x1_relu,inception_b1_1x7_3_relu,inception_b1_1x1_relu] )
inception_b2_1x1_2 = mx.symbol.Convolution(name='inception_b2_1x1_2', data=inception_b1_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b2_1x1_2_bn = mx.symbol.BatchNorm(name='inception_b2_1x1_2_bn', data=inception_b2_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_1x1_2_scale = inception_b2_1x1_2_bn
inception_b2_1x1_2_relu = mx.symbol.Activation(name='inception_b2_1x1_2_relu', data=inception_b2_1x1_2_scale , act_type='relu')
inception_b2_1x7_reduce = mx.symbol.Convolution(name='inception_b2_1x7_reduce', data=inception_b1_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b2_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_b2_1x7_reduce_bn', data=inception_b2_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_1x7_reduce_scale = inception_b2_1x7_reduce_bn
inception_b2_1x7_reduce_relu = mx.symbol.Activation(name='inception_b2_1x7_reduce_relu', data=inception_b2_1x7_reduce_scale , act_type='relu')
inception_b2_1x7 = mx.symbol.Convolution(name='inception_b2_1x7', data=inception_b2_1x7_reduce_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b2_1x7_bn = mx.symbol.BatchNorm(name='inception_b2_1x7_bn', data=inception_b2_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_1x7_scale = inception_b2_1x7_bn
inception_b2_1x7_relu = mx.symbol.Activation(name='inception_b2_1x7_relu', data=inception_b2_1x7_scale , act_type='relu')
inception_b2_7x1 = mx.symbol.Convolution(name='inception_b2_7x1', data=inception_b2_1x7_relu , num_filter=256, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b2_7x1_bn = mx.symbol.BatchNorm(name='inception_b2_7x1_bn', data=inception_b2_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_7x1_scale = inception_b2_7x1_bn
inception_b2_7x1_relu = mx.symbol.Activation(name='inception_b2_7x1_relu', data=inception_b2_7x1_scale , act_type='relu')
inception_b2_7x1_2_reduce = mx.symbol.Convolution(name='inception_b2_7x1_2_reduce', data=inception_b1_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b2_7x1_2_reduce_bn = mx.symbol.BatchNorm(name='inception_b2_7x1_2_reduce_bn', data=inception_b2_7x1_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_7x1_2_reduce_scale = inception_b2_7x1_2_reduce_bn
inception_b2_7x1_2_reduce_relu = mx.symbol.Activation(name='inception_b2_7x1_2_reduce_relu', data=inception_b2_7x1_2_reduce_scale , act_type='relu')
inception_b2_7x1_2 = mx.symbol.Convolution(name='inception_b2_7x1_2', data=inception_b2_7x1_2_reduce_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b2_7x1_2_bn = mx.symbol.BatchNorm(name='inception_b2_7x1_2_bn', data=inception_b2_7x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_7x1_2_scale = inception_b2_7x1_2_bn
inception_b2_7x1_2_relu = mx.symbol.Activation(name='inception_b2_7x1_2_relu', data=inception_b2_7x1_2_scale , act_type='relu')
inception_b2_1x7_2 = mx.symbol.Convolution(name='inception_b2_1x7_2', data=inception_b2_7x1_2_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b2_1x7_2_bn = mx.symbol.BatchNorm(name='inception_b2_1x7_2_bn', data=inception_b2_1x7_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_1x7_2_scale = inception_b2_1x7_2_bn
inception_b2_1x7_2_relu = mx.symbol.Activation(name='inception_b2_1x7_2_relu', data=inception_b2_1x7_2_scale , act_type='relu')
inception_b2_7x1_3 = mx.symbol.Convolution(name='inception_b2_7x1_3', data=inception_b2_1x7_2_relu , num_filter=224, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b2_7x1_3_bn = mx.symbol.BatchNorm(name='inception_b2_7x1_3_bn', data=inception_b2_7x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_7x1_3_scale = inception_b2_7x1_3_bn
inception_b2_7x1_3_relu = mx.symbol.Activation(name='inception_b2_7x1_3_relu', data=inception_b2_7x1_3_scale , act_type='relu')
inception_b2_1x7_3 = mx.symbol.Convolution(name='inception_b2_1x7_3', data=inception_b2_7x1_3_relu , num_filter=256, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b2_1x7_3_bn = mx.symbol.BatchNorm(name='inception_b2_1x7_3_bn', data=inception_b2_1x7_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_1x7_3_scale = inception_b2_1x7_3_bn
inception_b2_1x7_3_relu = mx.symbol.Activation(name='inception_b2_1x7_3_relu', data=inception_b2_1x7_3_scale , act_type='relu')
inception_b2_pool_ave = mx.symbol.Pooling(name='inception_b2_pool_ave', data=inception_b1_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_b2_1x1 = mx.symbol.Convolution(name='inception_b2_1x1', data=inception_b2_pool_ave , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b2_1x1_bn = mx.symbol.BatchNorm(name='inception_b2_1x1_bn', data=inception_b2_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b2_1x1_scale = inception_b2_1x1_bn
inception_b2_1x1_relu = mx.symbol.Activation(name='inception_b2_1x1_relu', data=inception_b2_1x1_scale , act_type='relu')
inception_b2_concat = mx.symbol.Concat(name='inception_b2_concat', *[inception_b2_1x1_2_relu,inception_b2_7x1_relu,inception_b2_1x7_3_relu,inception_b2_1x1_relu] )
inception_b3_1x1_2 = mx.symbol.Convolution(name='inception_b3_1x1_2', data=inception_b2_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b3_1x1_2_bn = mx.symbol.BatchNorm(name='inception_b3_1x1_2_bn', data=inception_b3_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_1x1_2_scale = inception_b3_1x1_2_bn
inception_b3_1x1_2_relu = mx.symbol.Activation(name='inception_b3_1x1_2_relu', data=inception_b3_1x1_2_scale , act_type='relu')
inception_b3_1x7_reduce = mx.symbol.Convolution(name='inception_b3_1x7_reduce', data=inception_b2_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b3_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_b3_1x7_reduce_bn', data=inception_b3_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_1x7_reduce_scale = inception_b3_1x7_reduce_bn
inception_b3_1x7_reduce_relu = mx.symbol.Activation(name='inception_b3_1x7_reduce_relu', data=inception_b3_1x7_reduce_scale , act_type='relu')
inception_b3_1x7 = mx.symbol.Convolution(name='inception_b3_1x7', data=inception_b3_1x7_reduce_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b3_1x7_bn = mx.symbol.BatchNorm(name='inception_b3_1x7_bn', data=inception_b3_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_1x7_scale = inception_b3_1x7_bn
inception_b3_1x7_relu = mx.symbol.Activation(name='inception_b3_1x7_relu', data=inception_b3_1x7_scale , act_type='relu')
inception_b3_7x1 = mx.symbol.Convolution(name='inception_b3_7x1', data=inception_b3_1x7_relu , num_filter=256, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b3_7x1_bn = mx.symbol.BatchNorm(name='inception_b3_7x1_bn', data=inception_b3_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_7x1_scale = inception_b3_7x1_bn
inception_b3_7x1_relu = mx.symbol.Activation(name='inception_b3_7x1_relu', data=inception_b3_7x1_scale , act_type='relu')
inception_b3_7x1_2_reduce = mx.symbol.Convolution(name='inception_b3_7x1_2_reduce', data=inception_b2_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b3_7x1_2_reduce_bn = mx.symbol.BatchNorm(name='inception_b3_7x1_2_reduce_bn', data=inception_b3_7x1_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_7x1_2_reduce_scale = inception_b3_7x1_2_reduce_bn
inception_b3_7x1_2_reduce_relu = mx.symbol.Activation(name='inception_b3_7x1_2_reduce_relu', data=inception_b3_7x1_2_reduce_scale , act_type='relu')
inception_b3_7x1_2 = mx.symbol.Convolution(name='inception_b3_7x1_2', data=inception_b3_7x1_2_reduce_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b3_7x1_2_bn = mx.symbol.BatchNorm(name='inception_b3_7x1_2_bn', data=inception_b3_7x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_7x1_2_scale = inception_b3_7x1_2_bn
inception_b3_7x1_2_relu = mx.symbol.Activation(name='inception_b3_7x1_2_relu', data=inception_b3_7x1_2_scale , act_type='relu')
inception_b3_1x7_2 = mx.symbol.Convolution(name='inception_b3_1x7_2', data=inception_b3_7x1_2_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b3_1x7_2_bn = mx.symbol.BatchNorm(name='inception_b3_1x7_2_bn', data=inception_b3_1x7_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_1x7_2_scale = inception_b3_1x7_2_bn
inception_b3_1x7_2_relu = mx.symbol.Activation(name='inception_b3_1x7_2_relu', data=inception_b3_1x7_2_scale , act_type='relu')
inception_b3_7x1_3 = mx.symbol.Convolution(name='inception_b3_7x1_3', data=inception_b3_1x7_2_relu , num_filter=224, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b3_7x1_3_bn = mx.symbol.BatchNorm(name='inception_b3_7x1_3_bn', data=inception_b3_7x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_7x1_3_scale = inception_b3_7x1_3_bn
inception_b3_7x1_3_relu = mx.symbol.Activation(name='inception_b3_7x1_3_relu', data=inception_b3_7x1_3_scale , act_type='relu')
inception_b3_1x7_3 = mx.symbol.Convolution(name='inception_b3_1x7_3', data=inception_b3_7x1_3_relu , num_filter=256, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b3_1x7_3_bn = mx.symbol.BatchNorm(name='inception_b3_1x7_3_bn', data=inception_b3_1x7_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_1x7_3_scale = inception_b3_1x7_3_bn
inception_b3_1x7_3_relu = mx.symbol.Activation(name='inception_b3_1x7_3_relu', data=inception_b3_1x7_3_scale , act_type='relu')
inception_b3_pool_ave = mx.symbol.Pooling(name='inception_b3_pool_ave', data=inception_b2_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_b3_1x1 = mx.symbol.Convolution(name='inception_b3_1x1', data=inception_b3_pool_ave , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b3_1x1_bn = mx.symbol.BatchNorm(name='inception_b3_1x1_bn', data=inception_b3_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b3_1x1_scale = inception_b3_1x1_bn
inception_b3_1x1_relu = mx.symbol.Activation(name='inception_b3_1x1_relu', data=inception_b3_1x1_scale , act_type='relu')
inception_b3_concat = mx.symbol.Concat(name='inception_b3_concat', *[inception_b3_1x1_2_relu,inception_b3_7x1_relu,inception_b3_1x7_3_relu,inception_b3_1x1_relu] )
inception_b4_1x1_2 = mx.symbol.Convolution(name='inception_b4_1x1_2', data=inception_b3_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b4_1x1_2_bn = mx.symbol.BatchNorm(name='inception_b4_1x1_2_bn', data=inception_b4_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_1x1_2_scale = inception_b4_1x1_2_bn
inception_b4_1x1_2_relu = mx.symbol.Activation(name='inception_b4_1x1_2_relu', data=inception_b4_1x1_2_scale , act_type='relu')
inception_b4_1x7_reduce = mx.symbol.Convolution(name='inception_b4_1x7_reduce', data=inception_b3_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b4_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_b4_1x7_reduce_bn', data=inception_b4_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_1x7_reduce_scale = inception_b4_1x7_reduce_bn
inception_b4_1x7_reduce_relu = mx.symbol.Activation(name='inception_b4_1x7_reduce_relu', data=inception_b4_1x7_reduce_scale , act_type='relu')
inception_b4_1x7 = mx.symbol.Convolution(name='inception_b4_1x7', data=inception_b4_1x7_reduce_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b4_1x7_bn = mx.symbol.BatchNorm(name='inception_b4_1x7_bn', data=inception_b4_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_1x7_scale = inception_b4_1x7_bn
inception_b4_1x7_relu = mx.symbol.Activation(name='inception_b4_1x7_relu', data=inception_b4_1x7_scale , act_type='relu')
inception_b4_7x1 = mx.symbol.Convolution(name='inception_b4_7x1', data=inception_b4_1x7_relu , num_filter=256, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b4_7x1_bn = mx.symbol.BatchNorm(name='inception_b4_7x1_bn', data=inception_b4_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_7x1_scale = inception_b4_7x1_bn
inception_b4_7x1_relu = mx.symbol.Activation(name='inception_b4_7x1_relu', data=inception_b4_7x1_scale , act_type='relu')
inception_b4_7x1_2_reduce = mx.symbol.Convolution(name='inception_b4_7x1_2_reduce', data=inception_b3_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b4_7x1_2_reduce_bn = mx.symbol.BatchNorm(name='inception_b4_7x1_2_reduce_bn', data=inception_b4_7x1_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_7x1_2_reduce_scale = inception_b4_7x1_2_reduce_bn
inception_b4_7x1_2_reduce_relu = mx.symbol.Activation(name='inception_b4_7x1_2_reduce_relu', data=inception_b4_7x1_2_reduce_scale , act_type='relu')
inception_b4_7x1_2 = mx.symbol.Convolution(name='inception_b4_7x1_2', data=inception_b4_7x1_2_reduce_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b4_7x1_2_bn = mx.symbol.BatchNorm(name='inception_b4_7x1_2_bn', data=inception_b4_7x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_7x1_2_scale = inception_b4_7x1_2_bn
inception_b4_7x1_2_relu = mx.symbol.Activation(name='inception_b4_7x1_2_relu', data=inception_b4_7x1_2_scale , act_type='relu')
inception_b4_1x7_2 = mx.symbol.Convolution(name='inception_b4_1x7_2', data=inception_b4_7x1_2_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b4_1x7_2_bn = mx.symbol.BatchNorm(name='inception_b4_1x7_2_bn', data=inception_b4_1x7_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_1x7_2_scale = inception_b4_1x7_2_bn
inception_b4_1x7_2_relu = mx.symbol.Activation(name='inception_b4_1x7_2_relu', data=inception_b4_1x7_2_scale , act_type='relu')
inception_b4_7x1_3 = mx.symbol.Convolution(name='inception_b4_7x1_3', data=inception_b4_1x7_2_relu , num_filter=224, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b4_7x1_3_bn = mx.symbol.BatchNorm(name='inception_b4_7x1_3_bn', data=inception_b4_7x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_7x1_3_scale = inception_b4_7x1_3_bn
inception_b4_7x1_3_relu = mx.symbol.Activation(name='inception_b4_7x1_3_relu', data=inception_b4_7x1_3_scale , act_type='relu')
inception_b4_1x7_3 = mx.symbol.Convolution(name='inception_b4_1x7_3', data=inception_b4_7x1_3_relu , num_filter=256, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b4_1x7_3_bn = mx.symbol.BatchNorm(name='inception_b4_1x7_3_bn', data=inception_b4_1x7_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_1x7_3_scale = inception_b4_1x7_3_bn
inception_b4_1x7_3_relu = mx.symbol.Activation(name='inception_b4_1x7_3_relu', data=inception_b4_1x7_3_scale , act_type='relu')
inception_b4_pool_ave = mx.symbol.Pooling(name='inception_b4_pool_ave', data=inception_b3_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_b4_1x1 = mx.symbol.Convolution(name='inception_b4_1x1', data=inception_b4_pool_ave , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b4_1x1_bn = mx.symbol.BatchNorm(name='inception_b4_1x1_bn', data=inception_b4_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b4_1x1_scale = inception_b4_1x1_bn
inception_b4_1x1_relu = mx.symbol.Activation(name='inception_b4_1x1_relu', data=inception_b4_1x1_scale , act_type='relu')
inception_b4_concat = mx.symbol.Concat(name='inception_b4_concat', *[inception_b4_1x1_2_relu,inception_b4_7x1_relu,inception_b4_1x7_3_relu,inception_b4_1x1_relu] )
inception_b5_1x1_2 = mx.symbol.Convolution(name='inception_b5_1x1_2', data=inception_b4_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b5_1x1_2_bn = mx.symbol.BatchNorm(name='inception_b5_1x1_2_bn', data=inception_b5_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_1x1_2_scale = inception_b5_1x1_2_bn
inception_b5_1x1_2_relu = mx.symbol.Activation(name='inception_b5_1x1_2_relu', data=inception_b5_1x1_2_scale , act_type='relu')
inception_b5_1x7_reduce = mx.symbol.Convolution(name='inception_b5_1x7_reduce', data=inception_b4_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b5_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_b5_1x7_reduce_bn', data=inception_b5_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_1x7_reduce_scale = inception_b5_1x7_reduce_bn
inception_b5_1x7_reduce_relu = mx.symbol.Activation(name='inception_b5_1x7_reduce_relu', data=inception_b5_1x7_reduce_scale , act_type='relu')
inception_b5_1x7 = mx.symbol.Convolution(name='inception_b5_1x7', data=inception_b5_1x7_reduce_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b5_1x7_bn = mx.symbol.BatchNorm(name='inception_b5_1x7_bn', data=inception_b5_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_1x7_scale = inception_b5_1x7_bn
inception_b5_1x7_relu = mx.symbol.Activation(name='inception_b5_1x7_relu', data=inception_b5_1x7_scale , act_type='relu')
inception_b5_7x1 = mx.symbol.Convolution(name='inception_b5_7x1', data=inception_b5_1x7_relu , num_filter=256, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b5_7x1_bn = mx.symbol.BatchNorm(name='inception_b5_7x1_bn', data=inception_b5_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_7x1_scale = inception_b5_7x1_bn
inception_b5_7x1_relu = mx.symbol.Activation(name='inception_b5_7x1_relu', data=inception_b5_7x1_scale , act_type='relu')
inception_b5_7x1_2_reduce = mx.symbol.Convolution(name='inception_b5_7x1_2_reduce', data=inception_b4_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b5_7x1_2_reduce_bn = mx.symbol.BatchNorm(name='inception_b5_7x1_2_reduce_bn', data=inception_b5_7x1_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_7x1_2_reduce_scale = inception_b5_7x1_2_reduce_bn
inception_b5_7x1_2_reduce_relu = mx.symbol.Activation(name='inception_b5_7x1_2_reduce_relu', data=inception_b5_7x1_2_reduce_scale , act_type='relu')
inception_b5_7x1_2 = mx.symbol.Convolution(name='inception_b5_7x1_2', data=inception_b5_7x1_2_reduce_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b5_7x1_2_bn = mx.symbol.BatchNorm(name='inception_b5_7x1_2_bn', data=inception_b5_7x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_7x1_2_scale = inception_b5_7x1_2_bn
inception_b5_7x1_2_relu = mx.symbol.Activation(name='inception_b5_7x1_2_relu', data=inception_b5_7x1_2_scale , act_type='relu')
inception_b5_1x7_2 = mx.symbol.Convolution(name='inception_b5_1x7_2', data=inception_b5_7x1_2_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b5_1x7_2_bn = mx.symbol.BatchNorm(name='inception_b5_1x7_2_bn', data=inception_b5_1x7_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_1x7_2_scale = inception_b5_1x7_2_bn
inception_b5_1x7_2_relu = mx.symbol.Activation(name='inception_b5_1x7_2_relu', data=inception_b5_1x7_2_scale , act_type='relu')
inception_b5_7x1_3 = mx.symbol.Convolution(name='inception_b5_7x1_3', data=inception_b5_1x7_2_relu , num_filter=224, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b5_7x1_3_bn = mx.symbol.BatchNorm(name='inception_b5_7x1_3_bn', data=inception_b5_7x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_7x1_3_scale = inception_b5_7x1_3_bn
inception_b5_7x1_3_relu = mx.symbol.Activation(name='inception_b5_7x1_3_relu', data=inception_b5_7x1_3_scale , act_type='relu')
inception_b5_1x7_3 = mx.symbol.Convolution(name='inception_b5_1x7_3', data=inception_b5_7x1_3_relu , num_filter=256, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b5_1x7_3_bn = mx.symbol.BatchNorm(name='inception_b5_1x7_3_bn', data=inception_b5_1x7_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_1x7_3_scale = inception_b5_1x7_3_bn
inception_b5_1x7_3_relu = mx.symbol.Activation(name='inception_b5_1x7_3_relu', data=inception_b5_1x7_3_scale , act_type='relu')
inception_b5_pool_ave = mx.symbol.Pooling(name='inception_b5_pool_ave', data=inception_b4_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_b5_1x1 = mx.symbol.Convolution(name='inception_b5_1x1', data=inception_b5_pool_ave , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b5_1x1_bn = mx.symbol.BatchNorm(name='inception_b5_1x1_bn', data=inception_b5_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b5_1x1_scale = inception_b5_1x1_bn
inception_b5_1x1_relu = mx.symbol.Activation(name='inception_b5_1x1_relu', data=inception_b5_1x1_scale , act_type='relu')
inception_b5_concat = mx.symbol.Concat(name='inception_b5_concat', *[inception_b5_1x1_2_relu,inception_b5_7x1_relu,inception_b5_1x7_3_relu,inception_b5_1x1_relu] )
inception_b6_1x1_2 = mx.symbol.Convolution(name='inception_b6_1x1_2', data=inception_b5_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b6_1x1_2_bn = mx.symbol.BatchNorm(name='inception_b6_1x1_2_bn', data=inception_b6_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_1x1_2_scale = inception_b6_1x1_2_bn
inception_b6_1x1_2_relu = mx.symbol.Activation(name='inception_b6_1x1_2_relu', data=inception_b6_1x1_2_scale , act_type='relu')
inception_b6_1x7_reduce = mx.symbol.Convolution(name='inception_b6_1x7_reduce', data=inception_b5_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b6_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_b6_1x7_reduce_bn', data=inception_b6_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_1x7_reduce_scale = inception_b6_1x7_reduce_bn
inception_b6_1x7_reduce_relu = mx.symbol.Activation(name='inception_b6_1x7_reduce_relu', data=inception_b6_1x7_reduce_scale , act_type='relu')
inception_b6_1x7 = mx.symbol.Convolution(name='inception_b6_1x7', data=inception_b6_1x7_reduce_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b6_1x7_bn = mx.symbol.BatchNorm(name='inception_b6_1x7_bn', data=inception_b6_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_1x7_scale = inception_b6_1x7_bn
inception_b6_1x7_relu = mx.symbol.Activation(name='inception_b6_1x7_relu', data=inception_b6_1x7_scale , act_type='relu')
inception_b6_7x1 = mx.symbol.Convolution(name='inception_b6_7x1', data=inception_b6_1x7_relu , num_filter=256, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b6_7x1_bn = mx.symbol.BatchNorm(name='inception_b6_7x1_bn', data=inception_b6_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_7x1_scale = inception_b6_7x1_bn
inception_b6_7x1_relu = mx.symbol.Activation(name='inception_b6_7x1_relu', data=inception_b6_7x1_scale , act_type='relu')
inception_b6_7x1_2_reduce = mx.symbol.Convolution(name='inception_b6_7x1_2_reduce', data=inception_b5_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b6_7x1_2_reduce_bn = mx.symbol.BatchNorm(name='inception_b6_7x1_2_reduce_bn', data=inception_b6_7x1_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_7x1_2_reduce_scale = inception_b6_7x1_2_reduce_bn
inception_b6_7x1_2_reduce_relu = mx.symbol.Activation(name='inception_b6_7x1_2_reduce_relu', data=inception_b6_7x1_2_reduce_scale , act_type='relu')
inception_b6_7x1_2 = mx.symbol.Convolution(name='inception_b6_7x1_2', data=inception_b6_7x1_2_reduce_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b6_7x1_2_bn = mx.symbol.BatchNorm(name='inception_b6_7x1_2_bn', data=inception_b6_7x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_7x1_2_scale = inception_b6_7x1_2_bn
inception_b6_7x1_2_relu = mx.symbol.Activation(name='inception_b6_7x1_2_relu', data=inception_b6_7x1_2_scale , act_type='relu')
inception_b6_1x7_2 = mx.symbol.Convolution(name='inception_b6_1x7_2', data=inception_b6_7x1_2_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b6_1x7_2_bn = mx.symbol.BatchNorm(name='inception_b6_1x7_2_bn', data=inception_b6_1x7_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_1x7_2_scale = inception_b6_1x7_2_bn
inception_b6_1x7_2_relu = mx.symbol.Activation(name='inception_b6_1x7_2_relu', data=inception_b6_1x7_2_scale , act_type='relu')
inception_b6_7x1_3 = mx.symbol.Convolution(name='inception_b6_7x1_3', data=inception_b6_1x7_2_relu , num_filter=224, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b6_7x1_3_bn = mx.symbol.BatchNorm(name='inception_b6_7x1_3_bn', data=inception_b6_7x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_7x1_3_scale = inception_b6_7x1_3_bn
inception_b6_7x1_3_relu = mx.symbol.Activation(name='inception_b6_7x1_3_relu', data=inception_b6_7x1_3_scale , act_type='relu')
inception_b6_1x7_3 = mx.symbol.Convolution(name='inception_b6_1x7_3', data=inception_b6_7x1_3_relu , num_filter=256, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b6_1x7_3_bn = mx.symbol.BatchNorm(name='inception_b6_1x7_3_bn', data=inception_b6_1x7_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_1x7_3_scale = inception_b6_1x7_3_bn
inception_b6_1x7_3_relu = mx.symbol.Activation(name='inception_b6_1x7_3_relu', data=inception_b6_1x7_3_scale , act_type='relu')
inception_b6_pool_ave = mx.symbol.Pooling(name='inception_b6_pool_ave', data=inception_b5_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_b6_1x1 = mx.symbol.Convolution(name='inception_b6_1x1', data=inception_b6_pool_ave , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b6_1x1_bn = mx.symbol.BatchNorm(name='inception_b6_1x1_bn', data=inception_b6_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b6_1x1_scale = inception_b6_1x1_bn
inception_b6_1x1_relu = mx.symbol.Activation(name='inception_b6_1x1_relu', data=inception_b6_1x1_scale , act_type='relu')
inception_b6_concat = mx.symbol.Concat(name='inception_b6_concat', *[inception_b6_1x1_2_relu,inception_b6_7x1_relu,inception_b6_1x7_3_relu,inception_b6_1x1_relu] )
inception_b7_1x1_2 = mx.symbol.Convolution(name='inception_b7_1x1_2', data=inception_b6_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b7_1x1_2_bn = mx.symbol.BatchNorm(name='inception_b7_1x1_2_bn', data=inception_b7_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_1x1_2_scale = inception_b7_1x1_2_bn
inception_b7_1x1_2_relu = mx.symbol.Activation(name='inception_b7_1x1_2_relu', data=inception_b7_1x1_2_scale , act_type='relu')
inception_b7_1x7_reduce = mx.symbol.Convolution(name='inception_b7_1x7_reduce', data=inception_b6_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b7_1x7_reduce_bn = mx.symbol.BatchNorm(name='inception_b7_1x7_reduce_bn', data=inception_b7_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_1x7_reduce_scale = inception_b7_1x7_reduce_bn
inception_b7_1x7_reduce_relu = mx.symbol.Activation(name='inception_b7_1x7_reduce_relu', data=inception_b7_1x7_reduce_scale , act_type='relu')
inception_b7_1x7 = mx.symbol.Convolution(name='inception_b7_1x7', data=inception_b7_1x7_reduce_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b7_1x7_bn = mx.symbol.BatchNorm(name='inception_b7_1x7_bn', data=inception_b7_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_1x7_scale = inception_b7_1x7_bn
inception_b7_1x7_relu = mx.symbol.Activation(name='inception_b7_1x7_relu', data=inception_b7_1x7_scale , act_type='relu')
inception_b7_7x1 = mx.symbol.Convolution(name='inception_b7_7x1', data=inception_b7_1x7_relu , num_filter=256, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b7_7x1_bn = mx.symbol.BatchNorm(name='inception_b7_7x1_bn', data=inception_b7_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_7x1_scale = inception_b7_7x1_bn
inception_b7_7x1_relu = mx.symbol.Activation(name='inception_b7_7x1_relu', data=inception_b7_7x1_scale , act_type='relu')
inception_b7_7x1_2_reduce = mx.symbol.Convolution(name='inception_b7_7x1_2_reduce', data=inception_b6_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b7_7x1_2_reduce_bn = mx.symbol.BatchNorm(name='inception_b7_7x1_2_reduce_bn', data=inception_b7_7x1_2_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_7x1_2_reduce_scale = inception_b7_7x1_2_reduce_bn
inception_b7_7x1_2_reduce_relu = mx.symbol.Activation(name='inception_b7_7x1_2_reduce_relu', data=inception_b7_7x1_2_reduce_scale , act_type='relu')
inception_b7_7x1_2 = mx.symbol.Convolution(name='inception_b7_7x1_2', data=inception_b7_7x1_2_reduce_relu , num_filter=192, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b7_7x1_2_bn = mx.symbol.BatchNorm(name='inception_b7_7x1_2_bn', data=inception_b7_7x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_7x1_2_scale = inception_b7_7x1_2_bn
inception_b7_7x1_2_relu = mx.symbol.Activation(name='inception_b7_7x1_2_relu', data=inception_b7_7x1_2_scale , act_type='relu')
inception_b7_1x7_2 = mx.symbol.Convolution(name='inception_b7_1x7_2', data=inception_b7_7x1_2_relu , num_filter=224, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b7_1x7_2_bn = mx.symbol.BatchNorm(name='inception_b7_1x7_2_bn', data=inception_b7_1x7_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_1x7_2_scale = inception_b7_1x7_2_bn
inception_b7_1x7_2_relu = mx.symbol.Activation(name='inception_b7_1x7_2_relu', data=inception_b7_1x7_2_scale , act_type='relu')
inception_b7_7x1_3 = mx.symbol.Convolution(name='inception_b7_7x1_3', data=inception_b7_1x7_2_relu , num_filter=224, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
inception_b7_7x1_3_bn = mx.symbol.BatchNorm(name='inception_b7_7x1_3_bn', data=inception_b7_7x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_7x1_3_scale = inception_b7_7x1_3_bn
inception_b7_7x1_3_relu = mx.symbol.Activation(name='inception_b7_7x1_3_relu', data=inception_b7_7x1_3_scale , act_type='relu')
inception_b7_1x7_3 = mx.symbol.Convolution(name='inception_b7_1x7_3', data=inception_b7_7x1_3_relu , num_filter=256, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
inception_b7_1x7_3_bn = mx.symbol.BatchNorm(name='inception_b7_1x7_3_bn', data=inception_b7_1x7_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_1x7_3_scale = inception_b7_1x7_3_bn
inception_b7_1x7_3_relu = mx.symbol.Activation(name='inception_b7_1x7_3_relu', data=inception_b7_1x7_3_scale , act_type='relu')
inception_b7_pool_ave = mx.symbol.Pooling(name='inception_b7_pool_ave', data=inception_b6_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_b7_1x1 = mx.symbol.Convolution(name='inception_b7_1x1', data=inception_b7_pool_ave , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_b7_1x1_bn = mx.symbol.BatchNorm(name='inception_b7_1x1_bn', data=inception_b7_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_b7_1x1_scale = inception_b7_1x1_bn
inception_b7_1x1_relu = mx.symbol.Activation(name='inception_b7_1x1_relu', data=inception_b7_1x1_scale , act_type='relu')
inception_b7_concat = mx.symbol.Concat(name='inception_b7_concat', *[inception_b7_1x1_2_relu,inception_b7_7x1_relu,inception_b7_1x7_3_relu,inception_b7_1x1_relu] )
reduction_b_3x3_reduce = mx.symbol.Convolution(name='reduction_b_3x3_reduce', data=inception_b7_concat , num_filter=192, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduction_b_3x3_reduce_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_reduce_bn', data=reduction_b_3x3_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_reduce_scale = reduction_b_3x3_reduce_bn
reduction_b_3x3_reduce_relu = mx.symbol.Activation(name='reduction_b_3x3_reduce_relu', data=reduction_b_3x3_reduce_scale , act_type='relu')
reduction_b_3x3 = mx.symbol.Convolution(name='reduction_b_3x3', data=reduction_b_3x3_reduce_relu , num_filter=192, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
reduction_b_3x3_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_bn', data=reduction_b_3x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_scale = reduction_b_3x3_bn
reduction_b_3x3_relu = mx.symbol.Activation(name='reduction_b_3x3_relu', data=reduction_b_3x3_scale , act_type='relu')
reduction_b_1x7_reduce = mx.symbol.Convolution(name='reduction_b_1x7_reduce', data=inception_b7_concat , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduction_b_1x7_reduce_bn = mx.symbol.BatchNorm(name='reduction_b_1x7_reduce_bn', data=reduction_b_1x7_reduce , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_1x7_reduce_scale = reduction_b_1x7_reduce_bn
reduction_b_1x7_reduce_relu = mx.symbol.Activation(name='reduction_b_1x7_reduce_relu', data=reduction_b_1x7_reduce_scale , act_type='relu')
reduction_b_1x7 = mx.symbol.Convolution(name='reduction_b_1x7', data=reduction_b_1x7_reduce_relu , num_filter=256, pad=(0, 3), kernel=(1,7), stride=(1,1), no_bias=True)
reduction_b_1x7_bn = mx.symbol.BatchNorm(name='reduction_b_1x7_bn', data=reduction_b_1x7 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_1x7_scale = reduction_b_1x7_bn
reduction_b_1x7_relu = mx.symbol.Activation(name='reduction_b_1x7_relu', data=reduction_b_1x7_scale , act_type='relu')
reduction_b_7x1 = mx.symbol.Convolution(name='reduction_b_7x1', data=reduction_b_1x7_relu , num_filter=320, pad=(3, 0), kernel=(7,1), stride=(1,1), no_bias=True)
reduction_b_7x1_bn = mx.symbol.BatchNorm(name='reduction_b_7x1_bn', data=reduction_b_7x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_7x1_scale = reduction_b_7x1_bn
reduction_b_7x1_relu = mx.symbol.Activation(name='reduction_b_7x1_relu', data=reduction_b_7x1_scale , act_type='relu')
reduction_b_3x3_2 = mx.symbol.Convolution(name='reduction_b_3x3_2', data=reduction_b_7x1_relu , num_filter=320, pad=(0, 0), kernel=(3,3), stride=(2,2), no_bias=True)
reduction_b_3x3_2_bn = mx.symbol.BatchNorm(name='reduction_b_3x3_2_bn', data=reduction_b_3x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
reduction_b_3x3_2_scale = reduction_b_3x3_2_bn
reduction_b_3x3_2_relu = mx.symbol.Activation(name='reduction_b_3x3_2_relu', data=reduction_b_3x3_2_scale , act_type='relu')
reduction_b_pool = mx.symbol.Pooling(name='reduction_b_pool', data=inception_b7_concat , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
reduction_b_concat = mx.symbol.Concat(name='reduction_b_concat', *[reduction_b_3x3_relu,reduction_b_3x3_2_relu,reduction_b_pool] )
inception_c1_1x1_2 = mx.symbol.Convolution(name='inception_c1_1x1_2', data=reduction_b_concat , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c1_1x1_2_bn = mx.symbol.BatchNorm(name='inception_c1_1x1_2_bn', data=inception_c1_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_1x1_2_scale = inception_c1_1x1_2_bn
inception_c1_1x1_2_relu = mx.symbol.Activation(name='inception_c1_1x1_2_relu', data=inception_c1_1x1_2_scale , act_type='relu')
inception_c1_1x1_3 = mx.symbol.Convolution(name='inception_c1_1x1_3', data=reduction_b_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c1_1x1_3_bn = mx.symbol.BatchNorm(name='inception_c1_1x1_3_bn', data=inception_c1_1x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_1x1_3_scale = inception_c1_1x1_3_bn
inception_c1_1x1_3_relu = mx.symbol.Activation(name='inception_c1_1x1_3_relu', data=inception_c1_1x1_3_scale , act_type='relu')
inception_c1_1x3 = mx.symbol.Convolution(name='inception_c1_1x3', data=inception_c1_1x1_3_relu , num_filter=256, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_c1_1x3_bn = mx.symbol.BatchNorm(name='inception_c1_1x3_bn', data=inception_c1_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_1x3_scale = inception_c1_1x3_bn
inception_c1_1x3_relu = mx.symbol.Activation(name='inception_c1_1x3_relu', data=inception_c1_1x3_scale , act_type='relu')
inception_c1_3x1 = mx.symbol.Convolution(name='inception_c1_3x1', data=inception_c1_1x1_3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_c1_3x1_bn = mx.symbol.BatchNorm(name='inception_c1_3x1_bn', data=inception_c1_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_3x1_scale = inception_c1_3x1_bn
inception_c1_3x1_relu = mx.symbol.Activation(name='inception_c1_3x1_relu', data=inception_c1_3x1_scale , act_type='relu')
inception_c1_1x1_4 = mx.symbol.Convolution(name='inception_c1_1x1_4', data=reduction_b_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c1_1x1_4_bn = mx.symbol.BatchNorm(name='inception_c1_1x1_4_bn', data=inception_c1_1x1_4 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_1x1_4_scale = inception_c1_1x1_4_bn
inception_c1_1x1_4_relu = mx.symbol.Activation(name='inception_c1_1x1_4_relu', data=inception_c1_1x1_4_scale , act_type='relu')
inception_c1_3x1_2 = mx.symbol.Convolution(name='inception_c1_3x1_2', data=inception_c1_1x1_4_relu , num_filter=448, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_c1_3x1_2_bn = mx.symbol.BatchNorm(name='inception_c1_3x1_2_bn', data=inception_c1_3x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_3x1_2_scale = inception_c1_3x1_2_bn
inception_c1_3x1_2_relu = mx.symbol.Activation(name='inception_c1_3x1_2_relu', data=inception_c1_3x1_2_scale , act_type='relu')
inception_c1_1x3_2 = mx.symbol.Convolution(name='inception_c1_1x3_2', data=inception_c1_3x1_2_relu , num_filter=512, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_c1_1x3_2_bn = mx.symbol.BatchNorm(name='inception_c1_1x3_2_bn', data=inception_c1_1x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_1x3_2_scale = inception_c1_1x3_2_bn
inception_c1_1x3_2_relu = mx.symbol.Activation(name='inception_c1_1x3_2_relu', data=inception_c1_1x3_2_scale , act_type='relu')
inception_c1_1x3_3 = mx.symbol.Convolution(name='inception_c1_1x3_3', data=inception_c1_1x3_2_relu , num_filter=256, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_c1_1x3_3_bn = mx.symbol.BatchNorm(name='inception_c1_1x3_3_bn', data=inception_c1_1x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_1x3_3_scale = inception_c1_1x3_3_bn
inception_c1_1x3_3_relu = mx.symbol.Activation(name='inception_c1_1x3_3_relu', data=inception_c1_1x3_3_scale , act_type='relu')
inception_c1_3x1_3 = mx.symbol.Convolution(name='inception_c1_3x1_3', data=inception_c1_1x3_2_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_c1_3x1_3_bn = mx.symbol.BatchNorm(name='inception_c1_3x1_3_bn', data=inception_c1_3x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_3x1_3_scale = inception_c1_3x1_3_bn
inception_c1_3x1_3_relu = mx.symbol.Activation(name='inception_c1_3x1_3_relu', data=inception_c1_3x1_3_scale , act_type='relu')
inception_c1_pool_ave = mx.symbol.Pooling(name='inception_c1_pool_ave', data=reduction_b_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_c1_1x1 = mx.symbol.Convolution(name='inception_c1_1x1', data=inception_c1_pool_ave , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c1_1x1_bn = mx.symbol.BatchNorm(name='inception_c1_1x1_bn', data=inception_c1_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c1_1x1_scale = inception_c1_1x1_bn
inception_c1_1x1_relu = mx.symbol.Activation(name='inception_c1_1x1_relu', data=inception_c1_1x1_scale , act_type='relu')
inception_c1_concat = mx.symbol.Concat(name='inception_c1_concat', *[inception_c1_1x1_2_relu,inception_c1_1x3_relu,inception_c1_3x1_relu,inception_c1_1x3_3_relu,inception_c1_3x1_3_relu,inception_c1_1x1_relu] )
inception_c2_1x1_2 = mx.symbol.Convolution(name='inception_c2_1x1_2', data=inception_c1_concat , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c2_1x1_2_bn = mx.symbol.BatchNorm(name='inception_c2_1x1_2_bn', data=inception_c2_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_1x1_2_scale = inception_c2_1x1_2_bn
inception_c2_1x1_2_relu = mx.symbol.Activation(name='inception_c2_1x1_2_relu', data=inception_c2_1x1_2_scale , act_type='relu')
inception_c2_1x1_3 = mx.symbol.Convolution(name='inception_c2_1x1_3', data=inception_c1_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c2_1x1_3_bn = mx.symbol.BatchNorm(name='inception_c2_1x1_3_bn', data=inception_c2_1x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_1x1_3_scale = inception_c2_1x1_3_bn
inception_c2_1x1_3_relu = mx.symbol.Activation(name='inception_c2_1x1_3_relu', data=inception_c2_1x1_3_scale , act_type='relu')
inception_c2_1x3 = mx.symbol.Convolution(name='inception_c2_1x3', data=inception_c2_1x1_3_relu , num_filter=256, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_c2_1x3_bn = mx.symbol.BatchNorm(name='inception_c2_1x3_bn', data=inception_c2_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_1x3_scale = inception_c2_1x3_bn
inception_c2_1x3_relu = mx.symbol.Activation(name='inception_c2_1x3_relu', data=inception_c2_1x3_scale , act_type='relu')
inception_c2_3x1 = mx.symbol.Convolution(name='inception_c2_3x1', data=inception_c2_1x1_3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_c2_3x1_bn = mx.symbol.BatchNorm(name='inception_c2_3x1_bn', data=inception_c2_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_3x1_scale = inception_c2_3x1_bn
inception_c2_3x1_relu = mx.symbol.Activation(name='inception_c2_3x1_relu', data=inception_c2_3x1_scale , act_type='relu')
inception_c2_1x1_4 = mx.symbol.Convolution(name='inception_c2_1x1_4', data=inception_c1_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c2_1x1_4_bn = mx.symbol.BatchNorm(name='inception_c2_1x1_4_bn', data=inception_c2_1x1_4 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_1x1_4_scale = inception_c2_1x1_4_bn
inception_c2_1x1_4_relu = mx.symbol.Activation(name='inception_c2_1x1_4_relu', data=inception_c2_1x1_4_scale , act_type='relu')
inception_c2_3x1_2 = mx.symbol.Convolution(name='inception_c2_3x1_2', data=inception_c2_1x1_4_relu , num_filter=448, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_c2_3x1_2_bn = mx.symbol.BatchNorm(name='inception_c2_3x1_2_bn', data=inception_c2_3x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_3x1_2_scale = inception_c2_3x1_2_bn
inception_c2_3x1_2_relu = mx.symbol.Activation(name='inception_c2_3x1_2_relu', data=inception_c2_3x1_2_scale , act_type='relu')
inception_c2_1x3_2 = mx.symbol.Convolution(name='inception_c2_1x3_2', data=inception_c2_3x1_2_relu , num_filter=512, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_c2_1x3_2_bn = mx.symbol.BatchNorm(name='inception_c2_1x3_2_bn', data=inception_c2_1x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_1x3_2_scale = inception_c2_1x3_2_bn
inception_c2_1x3_2_relu = mx.symbol.Activation(name='inception_c2_1x3_2_relu', data=inception_c2_1x3_2_scale , act_type='relu')
inception_c2_1x3_3 = mx.symbol.Convolution(name='inception_c2_1x3_3', data=inception_c2_1x3_2_relu , num_filter=256, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_c2_1x3_3_bn = mx.symbol.BatchNorm(name='inception_c2_1x3_3_bn', data=inception_c2_1x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_1x3_3_scale = inception_c2_1x3_3_bn
inception_c2_1x3_3_relu = mx.symbol.Activation(name='inception_c2_1x3_3_relu', data=inception_c2_1x3_3_scale , act_type='relu')
inception_c2_3x1_3 = mx.symbol.Convolution(name='inception_c2_3x1_3', data=inception_c2_1x3_2_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_c2_3x1_3_bn = mx.symbol.BatchNorm(name='inception_c2_3x1_3_bn', data=inception_c2_3x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_3x1_3_scale = inception_c2_3x1_3_bn
inception_c2_3x1_3_relu = mx.symbol.Activation(name='inception_c2_3x1_3_relu', data=inception_c2_3x1_3_scale , act_type='relu')
inception_c2_pool_ave = mx.symbol.Pooling(name='inception_c2_pool_ave', data=inception_c1_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_c2_1x1 = mx.symbol.Convolution(name='inception_c2_1x1', data=inception_c2_pool_ave , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c2_1x1_bn = mx.symbol.BatchNorm(name='inception_c2_1x1_bn', data=inception_c2_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c2_1x1_scale = inception_c2_1x1_bn
inception_c2_1x1_relu = mx.symbol.Activation(name='inception_c2_1x1_relu', data=inception_c2_1x1_scale , act_type='relu')
inception_c2_concat = mx.symbol.Concat(name='inception_c2_concat', *[inception_c2_1x1_2_relu,inception_c2_1x3_relu,inception_c2_3x1_relu,inception_c2_1x3_3_relu,inception_c2_3x1_3_relu,inception_c2_1x1_relu] )
inception_c3_1x1_2 = mx.symbol.Convolution(name='inception_c3_1x1_2', data=inception_c2_concat , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c3_1x1_2_bn = mx.symbol.BatchNorm(name='inception_c3_1x1_2_bn', data=inception_c3_1x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_1x1_2_scale = inception_c3_1x1_2_bn
inception_c3_1x1_2_relu = mx.symbol.Activation(name='inception_c3_1x1_2_relu', data=inception_c3_1x1_2_scale , act_type='relu')
inception_c3_1x1_3 = mx.symbol.Convolution(name='inception_c3_1x1_3', data=inception_c2_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c3_1x1_3_bn = mx.symbol.BatchNorm(name='inception_c3_1x1_3_bn', data=inception_c3_1x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_1x1_3_scale = inception_c3_1x1_3_bn
inception_c3_1x1_3_relu = mx.symbol.Activation(name='inception_c3_1x1_3_relu', data=inception_c3_1x1_3_scale , act_type='relu')
inception_c3_1x3 = mx.symbol.Convolution(name='inception_c3_1x3', data=inception_c3_1x1_3_relu , num_filter=256, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_c3_1x3_bn = mx.symbol.BatchNorm(name='inception_c3_1x3_bn', data=inception_c3_1x3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_1x3_scale = inception_c3_1x3_bn
inception_c3_1x3_relu = mx.symbol.Activation(name='inception_c3_1x3_relu', data=inception_c3_1x3_scale , act_type='relu')
inception_c3_3x1 = mx.symbol.Convolution(name='inception_c3_3x1', data=inception_c3_1x1_3_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_c3_3x1_bn = mx.symbol.BatchNorm(name='inception_c3_3x1_bn', data=inception_c3_3x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_3x1_scale = inception_c3_3x1_bn
inception_c3_3x1_relu = mx.symbol.Activation(name='inception_c3_3x1_relu', data=inception_c3_3x1_scale , act_type='relu')
inception_c3_1x1_4 = mx.symbol.Convolution(name='inception_c3_1x1_4', data=inception_c2_concat , num_filter=384, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c3_1x1_4_bn = mx.symbol.BatchNorm(name='inception_c3_1x1_4_bn', data=inception_c3_1x1_4 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_1x1_4_scale = inception_c3_1x1_4_bn
inception_c3_1x1_4_relu = mx.symbol.Activation(name='inception_c3_1x1_4_relu', data=inception_c3_1x1_4_scale , act_type='relu')
inception_c3_3x1_2 = mx.symbol.Convolution(name='inception_c3_3x1_2', data=inception_c3_1x1_4_relu , num_filter=448, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_c3_3x1_2_bn = mx.symbol.BatchNorm(name='inception_c3_3x1_2_bn', data=inception_c3_3x1_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_3x1_2_scale = inception_c3_3x1_2_bn
inception_c3_3x1_2_relu = mx.symbol.Activation(name='inception_c3_3x1_2_relu', data=inception_c3_3x1_2_scale , act_type='relu')
inception_c3_1x3_2 = mx.symbol.Convolution(name='inception_c3_1x3_2', data=inception_c3_3x1_2_relu , num_filter=512, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_c3_1x3_2_bn = mx.symbol.BatchNorm(name='inception_c3_1x3_2_bn', data=inception_c3_1x3_2 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_1x3_2_scale = inception_c3_1x3_2_bn
inception_c3_1x3_2_relu = mx.symbol.Activation(name='inception_c3_1x3_2_relu', data=inception_c3_1x3_2_scale , act_type='relu')
inception_c3_1x3_3 = mx.symbol.Convolution(name='inception_c3_1x3_3', data=inception_c3_1x3_2_relu , num_filter=256, pad=(0, 1), kernel=(1,3), stride=(1,1), no_bias=True)
inception_c3_1x3_3_bn = mx.symbol.BatchNorm(name='inception_c3_1x3_3_bn', data=inception_c3_1x3_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_1x3_3_scale = inception_c3_1x3_3_bn
inception_c3_1x3_3_relu = mx.symbol.Activation(name='inception_c3_1x3_3_relu', data=inception_c3_1x3_3_scale , act_type='relu')
inception_c3_3x1_3 = mx.symbol.Convolution(name='inception_c3_3x1_3', data=inception_c3_1x3_2_relu , num_filter=256, pad=(1, 0), kernel=(3,1), stride=(1,1), no_bias=True)
inception_c3_3x1_3_bn = mx.symbol.BatchNorm(name='inception_c3_3x1_3_bn', data=inception_c3_3x1_3 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_3x1_3_scale = inception_c3_3x1_3_bn
inception_c3_3x1_3_relu = mx.symbol.Activation(name='inception_c3_3x1_3_relu', data=inception_c3_3x1_3_scale , act_type='relu')
inception_c3_pool_ave = mx.symbol.Pooling(name='inception_c3_pool_ave', data=inception_c2_concat , pooling_convention='full', pad=(1,1), kernel=(3,3), stride=(1,1), pool_type='avg')
inception_c3_1x1 = mx.symbol.Convolution(name='inception_c3_1x1', data=inception_c3_pool_ave , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
inception_c3_1x1_bn = mx.symbol.BatchNorm(name='inception_c3_1x1_bn', data=inception_c3_1x1 , use_global_stats=True, fix_gamma=False, eps=0.001000)
inception_c3_1x1_scale = inception_c3_1x1_bn
inception_c3_1x1_relu = mx.symbol.Activation(name='inception_c3_1x1_relu', data=inception_c3_1x1_scale , act_type='relu')
inception_c3_concat = mx.symbol.Concat(name='inception_c3_concat', *[inception_c3_1x1_2_relu,inception_c3_1x3_relu,inception_c3_3x1_relu,inception_c3_1x3_3_relu,inception_c3_3x1_3_relu,inception_c3_1x1_relu] )
pool_8x8_s1 = mx.symbol.Pooling(name='pool_8x8_s1', data=inception_c3_concat , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
pool_8x8_s1_drop = mx.symbol.Dropout(name='pool_8x8_s1_drop', data=pool_8x8_s1 , p=0.200000)
flatten_0=mx.symbol.Flatten(name='flatten_0', data=pool_8x8_s1_drop)
classifier = mx.symbol.FullyConnected(name='classifier', data=flatten_0 , num_hidden=1000, no_bias=False)
prob = mx.symbol.SoftmaxOutput(name='prob', data=classifier )
