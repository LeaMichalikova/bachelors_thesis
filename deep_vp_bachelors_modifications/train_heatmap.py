import os

from tensorflow import keras

from models.hourglass import heatmap_mean_accuracy, load_model, parse_command_line, using_tiling
from datasets.heatmap_dataset import HeatmapBoxCarsDataset
from utils.gpu import set_gpus


def train():
    args = parse_command_line()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_gpus()

    model, scales, snapshot_dir_name, snapshot_dir_path = load_model(args)

    adam = keras.optimizers.Adam(args.learning_rate)

    num_vps = 2
    if using_tiling(args.modification):
        num_output_channels = num_vps
    else:
        num_output_channels = len(scales) * num_vps

    metric = heatmap_mean_accuracy(
        args.batch_size,
        args.heatmap_size,
        num_output_channels,
        args.modification
    )

    model.compile(
        adam, # = optimiser using gradient descend
        'MSE', # = L2 loss
        metrics=[metric, metric]
    )

    print(model.summary())

    print("Loading dataset!")
    train_dataset = HeatmapBoxCarsDataset(args.path, 'train', batch_size=args.batch_size, img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales, peak_original=args.peak_original, crop_delta=args.crop_delta, perspective_sigma=args.perspective_sigma, modif_num=args.modification)
    print("Loaded training dataset with {} samples".format(len(train_dataset.instance_list)))
    print("Using augmentation: ", args.perspective_sigma != 0.0 or args.crop_delta != 0)
    val_dataset = HeatmapBoxCarsDataset(args.path, 'val', batch_size=args.batch_size, img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales, peak_original=args.peak_original, modif_num=args.modification)
    print("Loaded val dataset with {} samples".format(len(val_dataset.instance_list)))

    callbacks = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(snapshot_dir_path, 'model.{epoch:03d}.h5')),
                 keras.callbacks.TensorBoard(log_dir=os.path.join('logs', snapshot_dir_name))]

    print("Workers: ", args.workers)
    print("Use multiprocessing: ", args.workers > 1)
    print("Starting training with lr: {}".format(args.learning_rate))

    model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=callbacks, initial_epoch=args.resume)

    if args.shutdown:
        os.system('sudo poweroff')


if __name__ == '__main__':
    train()
