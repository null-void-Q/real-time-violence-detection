"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.
 
The model is introduced in:
 
Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Joao Carreira, Andrew Zisserman
https://arxiv.org/abs/1705.07750v1
"""
        
from __future__ import print_function
from __future__ import absolute_import

import warnings

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D

from keras.utils.layer_utils import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K

WEIGHTS_NAME = ['rgb_inception_i3d', 'flow_inception_i3d','v_inception_i3d']

# path to pretrained models with top (classification layer)
WEIGHTS_PATH = {
    'rgb_inception_i3d': 'https://drive.google.com/uc?export=download&id=1JnqeLIM1izoccgz60fQJSWxuI00nf0_N',
    'flow_inception_i3d' : 'https://drive.google.com/uc?export=download&id=1YTik34v8jgPoN4UAF0k-g1rIqvwzuWrr',
    'v_inception_i3d':'https://drive.google.com/uc?export=download&id=1U9D0I8_CEcdLeupSYr7sdIZqVDxIFaZ6',
}

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP = {
    'rgb_inception_i3d': 'https://drive.google.com/uc?export=download&id=1l9GCEXncV8Izp905g6U-FMw01KKwLYga',
    'flow_inception_i3d' : 'https://drive.google.com/uc?export=download&id=1nRzNQYMgmPfv6OyYVBh10wa9rOEqwnfs'
}


def _obtain_input_shape(input_shape,
                        default_frame_size,
                        min_frame_size,
                        default_num_frames,
                        min_num_frames,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate the model's input shape.
    (Adapted from `keras/applications/imagenet_utils.py`)

    # Arguments
        input_shape: either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_frame_size: default input frames(images) width/height for the model.
        min_frame_size: minimum input frames(images) width/height accepted by the model.
        default_num_frames: default input number of frames(images) for the model.
        min_num_frames: minimum input number of frames accepted by the model.
        data_format: image data format to use.
        require_flatten: whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: one of `None` (random initialization)
            or 'kinetics (pre-training on Kinetics dataset with imagenet weights).
            If weights='kinetics' then
            input channels must be equal to 3.

    # Returns
        An integer shape tuple (may include None entries).

    # Raises
        ValueError: in case of invalid argument values.
    """
    if weights != 'kinetics' and input_shape and len(input_shape) == 4:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_num_frames, default_frame_size, default_frame_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_num_frames, default_frame_size, default_frame_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_num_frames, default_frame_size, default_frame_size)
        else:
            default_shape = (default_num_frames, default_frame_size, default_frame_size, 3)
    if (weights == 'kinetics') and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape

    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[0] != 3 and (weights == 'kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[1] is not None and input_shape[1] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[2] is not None and input_shape[2] < min_frame_size) or
                   (input_shape[3] is not None and input_shape[3] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[-1] != 3 and (weights == 'kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[0] is not None and input_shape[0] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[1] is not None and input_shape[1] < min_frame_size) or
                   (input_shape[2] is not None and input_shape[2] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None, None)
            else:
                input_shape = (None, None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


def conv3d_bn(x,
              filters,
              num_frames,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1, 1),
              use_bias = False,
              use_activation_fn = True,
              use_bn = True,
              name=None):
    """Utility function to apply conv3d + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not  
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(
        filters, (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=conv_name)(x)

    if use_bn:
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)

    return x


def Inception_Inflated3d(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                dropout_prob=0.0,
                endpoint_logit=True,
                classes=400, modelId = 'i3d'):
    """Instantiates the Inflated 3D Inception v1 architecture.

    Optionally loads weights pre-trained
    on Kinetics. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input frame(image) size for this model is 224x224.

    # Arguments
        include_top: whether to include the the classification 
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or 'kinetics' (pre-training on Kinetics dataset and imagenet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(NUM_FRAMES, 224, 224, 3)` (with `channels_last` data format)
            or `(NUM_FRAMES, 3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
            NUM_FRAMES should be no smaller than 8. The authors used 64
            frames per example for training and testing on kinetics dataset
            Also, Width and height should be no smaller than 32.
            E.g. `(64, 150, 150, 3)` would be one valid value.
        dropout_prob: optional, dropout probability applied in dropout layer
            after global average pooling layer. 
            0.0 means no dropout is applied, 1.0 means dropout is applied to all features.
            Note: Since Dropout is applied just before the classification
            layer, it is only useful when `include_top` is set to True.
        endpoint_logit: (boolean) optional. If True, the model's forward pass
            will end at producing logits. Otherwise, softmax is applied after producing
            the logits to produce the class probabilities prediction. Setting this parameter 
            to True is particularly useful when you want to combine results of rgb model
            and optical flow model.
            - `True` end model forward pass at logit output
            - `False` go further after logit to produce softmax predictions
            Note: This parameter is only useful when `include_top` is set to True.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights == WEIGHTS_NAME[2] and not include_top:
        raise ValueError('The Violence model cant be instatiated without top')        
    if not (weights in WEIGHTS_NAME or weights is None or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or %s' % 
                         str(WEIGHTS_NAME) + ' ' 
                         'or a valid path to a file containing `weights` values')

    if weights in WEIGHTS_NAME and include_top and (classes != 400 and classes != 2):
        raise ValueError('If using `weights` as one of these %s, with `include_top`'
                         ' as true, `classes` should be 400 or 2' % str(WEIGHTS_NAME))

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_frame_size=224, 
        min_frame_size=32, 
        default_num_frames=64,
        min_num_frames=8,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4

    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(img_input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,)

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    if include_top:
        # Classification block
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid')(x)
        x = Dropout(dropout_prob)(x)

        x = conv3d_bn(x, classes, 1, 1, 1, padding='same', 
                use_bias=True, use_activation_fn=False, use_bn=False)
 
        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, classes))(x)

        # logits (raw scores for each class)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        if not endpoint_logit:
            x = Activation('softmax')(x)
    else:
        h = int(x.shape[2])
        w = int(x.shape[3])
        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid')(x)



    inputs = img_input
    # create model
    model = Model(inputs, x)

    # load weights
    if weights in WEIGHTS_NAME:
        if weights == WEIGHTS_NAME[0]:   # rgb_kinetics
            if include_top:
                weights_url = WEIGHTS_PATH['rgb_inception_i3d']
                model_name = 'i3d_inception_rgb.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['rgb_inception_i3d']
                model_name = 'i3d_inception_rgb_no_top.h5'

        elif weights == WEIGHTS_NAME[1]: # flow_kinetics
            if include_top:
                weights_url = WEIGHTS_PATH['flow_inception_i3d']
                model_name = 'i3d_inception_flow.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['flow_inception_i3d']
                model_name = 'i3d_inception_flow_no_top.h5'
        elif weights == WEIGHTS_NAME[2]: # violence_model
            
            weights_url = WEIGHTS_PATH['v_inception_i3d']
            model_name = 'v_inception_i3d.h5'               

        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
        model.load_weights(downloaded_weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your keras config '
                          'at ~/.keras/keras.json.')

    elif weights is not None:
        model.load_weights(weights)

    return model
