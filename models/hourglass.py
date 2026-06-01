import argparse
import os
import string

from tensorflow import keras
import tensorflow as tf

from datetime import datetime

# Adapted from: https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/


def running_modification_number(current_modif_num, *desired_modif_num):
    return current_modif_num in desired_modif_num

def using_bilinear(current_modif_num):
    return running_modification_number(current_modif_num, 1, 2, 3, 4, 5)

def using_refinement_head(current_modif_num):
    return running_modification_number(current_modif_num, 1, 2)

def post_upsampling(current_modif_num):
    return running_modification_number(current_modif_num, 1)

def pre_upsampling(current_modif_num):
    return running_modification_number(current_modif_num, 2)

def using_tiling(current_modif_num):
    return running_modification_number(current_modif_num, 3)

def using_quadrant_upsampling(current_modif_num):
    return running_modification_number(current_modif_num, 4)

def using_channel_tiling(current_modif_num):
    return running_modification_number(current_modif_num, 5)


# https://www.tensorflow.org/api_docs/python/tf/keras/Layer
class ScaleFeatureTiler(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = 4 # always using four scales when tiling

    def call(self, inputs):
        ones = tf.ones_like(inputs[..., :1])
        zeros = tf.zeros_like(inputs[..., :1])
        tiles = []
        for i in range(self.num_scales):
            scale_channels = []
            for j in range(self.num_scales):
                if i == j:
                    scale_channels.append(ones)
                else:
                    scale_channels.append(zeros)
            scale_code = tf.concat(scale_channels, axis=-1)
            tiles.append(tf.concat([inputs, scale_code], axis=-1))

        top_tiles = tf.concat([tiles[0], tiles[1]], axis=2)
        bottom_tiles = tf.concat([tiles[2], tiles[3]], axis=2)
        return tf.concat([top_tiles, bottom_tiles], axis=1)

    def compute_output_shape(self, input_shape):
        # original: B, H, W, C
        # tiled: B, 2H, 2W, C+4
        return input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3] + self.num_scales

    def get_config(self):
        config = super().get_config()
        config.update({'num_scales': self.num_scales})
        return config

def heatmap_mean_accuracy(batch_size, heatmap_size, scale_size, current_modif_num):
    if using_tiling(current_modif_num):
        def mean_acc(y_pred, y_gt):
            h = w = heatmap_size
            quadrants = [(0, h, 0, w), (0, h, w, 2*w), (h, 2*h, 0, w), (h, 2*h, w, 2*w)]
            quadrants_eq = []
            for y0, y1, x0, x1 in quadrants:
                pred_tile = y_pred[:, y0:y1, x0:x1, :]
                gt_tile = y_gt[:, y0:y1, x0:x1, :]

                argmax_pred = tf.argmax(tf.reshape(pred_tile, [batch_size, heatmap_size * heatmap_size, scale_size]), axis=1)
                argmax_gt = tf.argmax(tf.reshape(gt_tile, [batch_size, heatmap_size * heatmap_size, scale_size]), axis=1)
                eq = tf.cast(tf.equal(argmax_pred, argmax_gt), "float32")
                quadrants_eq.append(eq)

            return keras.backend.mean(tf.concat(quadrants_eq, axis=1))

    else:
        def mean_acc(y_pred, y_gt):
            argmax_pred = tf.argmax(tf.reshape(y_pred, [batch_size, heatmap_size * heatmap_size, scale_size]), axis=1)
            argmax_gt = tf.argmax(tf.reshape(y_gt, [batch_size, heatmap_size * heatmap_size, scale_size]), axis=1)
            eq = tf.cast(tf.equal(argmax_pred, argmax_gt), "float32")
            return keras.backend.mean(eq)

    return mean_acc

def create_hourglass_network(num_classes, num_stacks, inres=128, outres=128, num_channels=256, bottleneck='bottleneck', modif_num=0):
    number_of_channels = 3 # one channel for each RGB color, ... 3 numbers per pixel (RGB) create an image
    input = keras.layers.Input(shape=(inres, inres, number_of_channels))
    inres_outres_ratio = round(inres / outres, 2)

    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}: inres_outres_ratio is {inres_outres_ratio}", flush=True)

    # checks for correct input-output ratios
    if (using_quadrant_upsampling(modif_num) or using_channel_tiling(modif_num)) and inres_outres_ratio != 0.5:
        raise ValueError("this modification requires the output resolution to be twice as big as the input resolution")
    elif using_tiling(modif_num) and inres_outres_ratio < 1:
        raise ValueError("producing bigger heatmaps is not supported for tiled heatmaps")
        # raise ValueError("tiled heatmaps require the output resolution to be the same as input resolution")

    if bottleneck == 'mobilenet':
        bottleneck = bottleneck_mobile
    else:
        bottleneck = bottleneck_block

    front_features = create_front_module(input, num_channels, bottleneck, inres_outres_ratio, modif_num=modif_num)

    head_next_stage = front_features

    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i, modif_num=modif_num)

        # 1st modification
        if post_upsampling(modif_num):
            current_ratio = inres_outres_ratio
            while current_ratio < 1:
                head_to_loss = keras.layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), activation='linear', padding='same',
                                                    name=str(i) + '_final_upsampling_transposition_x1')(head_to_loss)
                head_to_loss = keras.layers.Conv2D(num_classes, kernel_size=(3, 3), activation='linear', padding='same',
                                                    name=str(i) + '_final_upsampling_transposition_x2')(head_to_loss)
                current_ratio *= 2

        outputs.append(head_to_loss)

    model = keras.models.Model(inputs=input, outputs=outputs)
    # rms = RMSprop(lr=5e-4)
    # model.compile(optimizer=rms, loss=mean_squared_error, metrics=["accuracy"])
    return model


def hourglass_module(bottom, num_classes, num_channels, bottleneck, hgid, modif_num):
    # create left features , f1, f2, f4, and f8
    left_features = create_left_half_blocks(bottom, bottleneck, hgid, num_channels)

    # create right features, connect with left features
    rf1 = create_right_half_blocks(left_features, bottleneck, hgid, num_channels, modif_num=modif_num)

    # add 1x1 conv with two heads, head_next_stage is sent to next stage
    # head_parts is used for intermediate supervision
    head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hgid, num_channels, modif_num=modif_num)

    return head_next_stage, head_parts


def bottleneck_block(bottom, num_out_channels, block_name):
    # skip layer
    if keras.backend.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = keras.layers.Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                       name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = keras.layers.Conv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x1')(bottom)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.Conv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                name=block_name + '_conv_3x3_x2')(_x)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x3')(_x)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.Add(name=block_name + '_residual')([_skip, _x])

    return _x


def bottleneck_mobile(bottom, num_out_channels, block_name):
    # skip layer
    if keras.backend.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = keras.layers.SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                                name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = keras.layers.SeparableConv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_x1')(bottom)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.SeparableConv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                         name=block_name + '_conv_3x3_x2')(_x)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_x3')(_x)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.Add(name=block_name + '_residual')([_skip, _x])

    return _x


def create_front_module(input, num_channels, bottleneck, inres_outres_ratio=1, modif_num=0):
    # front module, input to 1/4 resolution
    # 1 7x7 conv + maxpooling
    # 3 residual block

    # 64 = channels = the layer will learn 64 different features, input was 128 128 3 but output will be 128 128 64
    # each channel exists for one learned feature (edge, corner, texture, ...) - NVM, WRONG

    number_of_filters = 64
    # kernel size = size of the filter, bigger kernel sees more context
    # strides = how far the filter moves each step, 2x2 means it skips pixels - reduces resolution by half
    # padding='same' = keeps spatial size (unless stride reduces it)
    # activation='relu' = turns negative numbers into 0 - introduces non-linearity, helps network learn complex patterns
    # maxpool2D = takes max value in each 2×2 region - keeps strongest features, removes noise, reduces computation

    if inres_outres_ratio == 4:
        _x = keras.layers.Conv2D(number_of_filters, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1_x1')(input)
        _x = keras.layers.BatchNormalization()(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x1')
        _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x2')
        _x = bottleneck(_x, num_channels, 'front_residual_x3')

    elif inres_outres_ratio == 2:
        _x = keras.layers.Conv2D(number_of_filters, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', name='front_conv_1x1_x1')(input)
        _x = keras.layers.BatchNormalization()(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x1')
        _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x2')
        _x = bottleneck(_x, num_channels, 'front_residual_x3')

    # 2nd modification
    elif pre_upsampling(modif_num):
        if inres_outres_ratio == 0.5:
            _x = keras.layers.Conv2D(number_of_filters, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', name='front_conv_1x1_x1')(input)
            _x = keras.layers.BatchNormalization()(_x)
            _x = bottleneck(_x, num_channels // 2, 'front_residual_x1')
            _x = keras.layers.Conv2DTranspose(num_channels, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu', name='front_conv2Dtranspose_x1')(_x)
            _x = bottleneck(_x, num_channels // 2, 'front_residual_x2')
            _x = bottleneck(_x, num_channels, 'front_residual_x3')
        else:
            raise ValueError("pre-upsampling requires the output resolution to be 2x larger than the input resolution")

    else:
        _x = keras.layers.Conv2D(number_of_filters, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', name='front_conv_1x1_x1')(input)
        _x = keras.layers.BatchNormalization()(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x1')
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x2')
        _x = bottleneck(_x, num_channels, 'front_residual_x3')

    return _x


def create_left_half_blocks(bottom, bottleneck, hglayer, num_channels):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    hgname = 'hg' + str(hglayer)

    f1 = bottleneck(bottom, num_channels, hgname + '_l1')
    _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

    f2 = bottleneck(_x, num_channels, hgname + '_l2')
    _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

    f4 = bottleneck(_x, num_channels, hgname + '_l4')
    _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

    f8 = bottleneck(_x, num_channels, hgname + '_l8')

    return (f1, f2, f4, f8)


def connect_left_to_right(left, right, bottleneck, name, num_channels, modif_num):
    '''
    :param left: connect left feature to right feature
    :param name: layer name
    :return:
    '''
    # left -> 1 bottlenect
    # right -> upsampling
    # keras.layers.Add   -> left + right

    _xleft = bottleneck(left, num_channels, name + '_connect')

    # modification
    if using_bilinear(modif_num):
        interpolation='bilinear'
    else:
        interpolation='nearest'

    _xright = keras.layers.UpSampling2D(interpolation=interpolation)(right)

    add = keras.layers.Add()([_xleft, _xright])
    out = bottleneck(add, num_channels, name + '_connect_conv')
    return out


def bottom_layer(lf8, bottleneck, hgid, num_channels):
    # blocks in lowest resolution
    # 3 bottlenect blocks + keras.layers.Add

    lf8_connect = bottleneck(lf8, num_channels, str(hgid) + "_lf8")

    _x = bottleneck(lf8, num_channels, str(hgid) + "_lf8_x1")
    _x = bottleneck(_x, num_channels, str(hgid) + "_lf8_x2")
    _x = bottleneck(_x, num_channels, str(hgid) + "_lf8_x3")

    rf8 = keras.layers.Add()([_x, lf8_connect])

    return rf8


def create_right_half_blocks(leftfeatures, bottleneck, hglayer, num_channels, modif_num):
    lf1, lf2, lf4, lf8 = leftfeatures

    rf8 = bottom_layer(lf8, bottleneck, hglayer, num_channels)

    rf4 = connect_left_to_right(lf4, rf8, bottleneck, 'hg' + str(hglayer) + '_rf4', num_channels, modif_num=modif_num)

    rf2 = connect_left_to_right(lf2, rf4, bottleneck, 'hg' + str(hglayer) + '_rf2', num_channels, modif_num=modif_num)

    rf1 = connect_left_to_right(lf1, rf2, bottleneck, 'hg' + str(hglayer) + '_rf1', num_channels, modif_num=modif_num)

    return rf1

def create_heads(prelayerfeatures, rf1, num_classes, hgid, num_channels, modif_num):
    # two head, one head to next stage, one head to intermediate features
    head = keras.layers.Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same',
                               name=str(hgid) + '_conv_1x1_x1')(rf1)
    head = keras.layers.BatchNormalization()(head)

    # part of 5th modification
    if using_channel_tiling(modif_num):
        num_quadrants = 4
        num_channels_tiling = num_classes * num_quadrants
        head_channel_tiling = keras.layers.Conv2D(num_channels_tiling, kernel_size=(1, 1), activation='linear', padding='same',
                                     name=str(hgid) + '_pre_channel_tiling')(head)

    # for head as intermediate supervision, use 'linear' as activation.
    head_parts = keras.layers.Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                                     name=str(hgid) + '_out')(head)

    # 1st and 2nd modification
    if using_refinement_head(modif_num):
        head_parts_decoder = keras.layers.Conv2D(num_channels, kernel_size=(3, 3), activation='relu', padding='same',
                                                 name=str(hgid) + '_out_decoder_x1')(head)
        head_parts_decoder = keras.layers.BatchNormalization()(head_parts_decoder)
        head_parts_decoder = keras.layers.Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                                                 name=str(hgid) + '_out_decoder_x2')(head_parts_decoder)

        head_parts_out = keras.layers.Add(name=str(hgid) + '_out_refined')([head_parts, head_parts_decoder])

    # 3rd modification
    elif using_tiling(modif_num):
        tiled_features = ScaleFeatureTiler(name=str(hgid) + '_scale_feature_tiler')(head)

        # inspired by bottleneck_block()
        head_tiled = keras.layers.Conv2D(num_channels, kernel_size=(3, 3), activation='relu', padding='same',
                                         name=str(hgid) + '_shared_tiled_x1')(tiled_features)
        head_tiled = keras.layers.BatchNormalization()(head_tiled)
        head_tiled = keras.layers.Conv2D(num_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                                         name=str(hgid) + '_shared_tiled_x2')(head_tiled)
        head_tiled = keras.layers.BatchNormalization()(head_tiled)

        # hardcoded 2 classes instead of 8 - corresponding to # of output heatmaps
        head_parts_out = keras.layers.Conv2D(2, kernel_size=(1, 1), activation='linear', padding='same',
                                             name=str(hgid) + '_out_shared_tiled')(head_tiled)

    # 4th modification
    elif using_quadrant_upsampling(modif_num):
        args = parse_command_line()
        # the size of one tile
        # one non-upsampled tile is half the size of an upsampled tile, which is half the size of the final heatmap
        t = args.heatmap_size // 4

        quadrants = [
            head_parts[:, 0:t, 0:t, :],
            head_parts[:, 0:t, t:2*t, :],
            head_parts[:, t:2*t, 0:t, :],
            head_parts[:, t:2*t, t:2*t, :]
        ]

        upsampled_quadrants = []
        for q_id, quadrant in enumerate(quadrants):
            quadrant = keras.layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), activation='relu',
                                                    padding='same', name=str(hgid) + 'quadrant' + str(q_id) + '_sliced_tile')(quadrant)
            upsampled_quadrants.append(quadrant)

        top_quads = keras.layers.Concatenate(axis=2)([upsampled_quadrants[0], upsampled_quadrants[1]])
        bottom_quads = keras.layers.Concatenate(axis=2)([upsampled_quadrants[2], upsampled_quadrants[3]])
        all_quads = keras.layers.Concatenate(axis=1)([top_quads, bottom_quads])

        head_parts_out = keras.layers.Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                                             name=str(hgid) + '_out_sliced_tiled')(all_quads)

    # 5th modification
    elif using_channel_tiling(modif_num):
        top_quads = keras.layers.Concatenate(axis=2)([head_channel_tiling[:, :, :, 0::4], head_channel_tiling[:, :, :, 1::4]])
        bottom_quads = keras.layers.Concatenate(axis=2)([head_channel_tiling[:, :, :, 2::4], head_channel_tiling[:, :, :, 3::4]])
        head_parts_out = keras.layers.Concatenate(axis=1, name=str(hgid) + '_out_channel_tiling')([top_quads, bottom_quads])

    # use linear activation
    head = keras.layers.Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                               name=str(hgid) + '_conv_1x1_x2')(head)
    head_m = keras.layers.Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                                 name=str(hgid) + '_conv_1x1_x3')(head_parts)

    head_next_stage = keras.layers.Add()([head, head_m, prelayerfeatures])

    # if not running any 2026 modifications
    if running_modification_number(modif_num, 0):
        return head_next_stage, head_parts
    else:
        return head_next_stage, head_parts_out

def load_model(args):
    print("Initializing model")
    print("Batch size: ", args.batch_size)
    print("Num stacks: ", args.num_stacks)
    print("Input size: {} x {}".format(args.input_size, args.input_size))
    print("Heatmap size: {} x {}".format(args.heatmap_size, args.heatmap_size))
    print("Training for {} epochs".format(args.epochs))
    print("Channels: ", args.channels)
    print("Experiment number: ", args.experiment)
    print("Mobilenet version: ", args.mobilenet)
    print("Heatmap distribution constructed in original coords: ", args.peak_original)

    if args.modification == 0:
        print("Not running any 2026 modifications")
    else:
        print("2026 modification number: ", args.modification)

    if len(args.scales) == 0:
        scales = [0.03, 0.1, 0.3, 1]
        scales_str = '4'
    else:
        scales = [float(x) for x in args.scales]
        scales_str = '-'.join(args.scales)

    # if we are running the 3rd modification (tiling heatmaps)
    if args.modification == 3:
        if len(args.scales) != 4:
            raise ValueError("tiled heatmaps require 4 scales")
        # sort the scales for correct quadrant assignment later
        scales = [float(x) for x in sorted(scales)]

    if args.mobilenet:
        module = 'mobilenet'
        module_str = 'm'
    else:
        module = 'bottleneck'
        module_str = 'b'

    peak_str = 'po' if args.peak_original else 'pd'

    if args.crop_delta == 0 and args.perspective_sigma == 0.0:
        print("Not using augmentation")

        aug_str ='noaug'
    else:
        print("Using augmentation")
        print("Perspectve sigma: {}".format(args.perspective_sigma))
        print("Crop delta: {}".format(args.crop_delta))
        aug_str = 'aug_{}ps_{}cd'.format(args.perspective_sigma, args.crop_delta)


    snapshot_dir_name = 'VP1VP2{}_{}_{}_{}in_{}out_{}s_{}n_{}b_{}c_{}'.format(module_str, peak_str, aug_str, args.input_size, args.heatmap_size,
                                                                        scales_str, args.num_stacks, args.batch_size,
                                                                        args.channels, args.experiment)
    snapshot_dir_path = os.path.join('snapshots', snapshot_dir_name)

    if not os.path.exists(snapshot_dir_path):
        os.makedirs(snapshot_dir_path)

    print("Checkpoint dir name: ", snapshot_dir_name)

    model = create_hourglass_network(2 * len(scales), args.num_stacks, inres=args.input_size, outres=args.heatmap_size,
                                     bottleneck=module, num_channels=args.channels, modif_num=args.modification)

    if args.resume:
        if args.experiment_resume is None:
            args.experiment_resume = args.experiment

        resume_dir_name = 'VP1VP2{}_{}_{}_{}in_{}out_{}s_{}n_{}b_{}c_{}'.format(module_str, peak_str, aug_str, args.input_size,
                                                                               args.heatmap_size,
                                                                               scales_str, args.num_stacks,
                                                                               args.batch_size,
                                                                               args.channels, args.experiment_resume)

        resume_model_path = os.path.join('snapshots', resume_dir_name, 'model.{:03d}.h5'.format(args.resume))
        print("Loading model", resume_model_path)
        model.load_weights(resume_model_path)

    return model, scales, snapshot_dir_name, snapshot_dir_path


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mod', '--modification', type=int, default=0, help='which modification from the 2026 work should be executed, 0 if running the non-modified version')
    parser.add_argument('-r', '--resume', type=int, default=0, help='resume from saved epoch snapshot')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-bb', '--batch_size_eval', type=int, default=4, help='batch size for evaluation scripts')
    parser.add_argument('-n', '--num_stacks', type=int, default=2, help='number of stacks')
    parser.add_argument('-i', '--input_size', type=int, default=128, help='size of input')
    parser.add_argument('-o', '--heatmap_size', type=int, default=64, help='size of output heatmaps')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('-m', '--mobilenet', action='store_true', default=False)
    parser.add_argument('-s', '--scales', nargs='*', action='store', default=[])
    parser.add_argument('-ps', '--perspective_sigma', type=float, default=25.0, help='perspective sigma for augmentation')
    parser.add_argument('-cd', '--crop_delta', type=int, default=10, help='crop delta for augmentation')
    parser.add_argument('-po', '--peak_original', action='store_true', default=False, help='whether to construct the peak in the original space')
    parser.add_argument('--shutdown', action='store_true', default=False, help='shutdown the machine when done')
    parser.add_argument('-c', '--channels', type=int, default=256, help='number of channels in network')
    parser.add_argument('-exp', '--experiment', type=int, default=0, help='experiment number')
    parser.add_argument('-expr', '--experiment-resume', type=int, default=None, help='experiment number to resume from')
    parser.add_argument('-w', '--workers', type=int, default=1, help='number of workers for the fit function')
    parser.add_argument('--debug', action='store_true', default=False, help='enable debug where applicable')
    parser.add_argument('-de', '--dump_every', type=int, default=0, help='save every n frames during extraction scripts')
    parser.add_argument('-mf', '--max_frames', type=int, default=5000, help='number of max frames to process during extraction')
    parser.add_argument('--mask', action='store_true', default=False, help='whether to use mask information during extraction')
    parser.add_argument('--skip', type=int, default=1, help='how many frames to skip in BrnoCompSpeed during extraction')
    parser.add_argument('--resize_imshow_frame_into', type=tuple, default=(800, 600),
                        help='this is for resizing the resulted image')
    # parser.add_argument('-s', '--steps', type=int, default=10000, help='steps per epoch')
    parser.add_argument('path')
    args = parser.parse_args()
    return args