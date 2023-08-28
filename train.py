from mmengine.runner import Runner
from mmengine.config import Config

from ship_dataset import ShipDataset


def main():
    config_path = "configs/rotated_rtmdet/rotated_rtmdet_l-100e-aug-ship.py"
    dataset_type = "ShipDataset"
    data_root_path = "data/ssdd_tiny/"
    working_dir = "./runs"

    batch_size = 4
    epochs = 15

    # Load
    config = Config.fromfile(config_path)
    print(f"Loaded config from path: {config}...")

    # Modify dataset type and path
    # => dataset & dataloader
    config.dataset_type = dataset_type
    config.data_root = data_root_path

    config.train_dataloader.dataset.type = dataset_type
    config.train_dataloader.dataset.ann_file = 'train'
    config.train_dataloader.dataset.data_prefix.img_path = 'images'
    config.train_dataloader.dataset.data_root = data_root_path

    config.val_dataloader.dataset.type = dataset_type
    config.val_dataloader.dataset.ann_file = 'val'
    config.val_dataloader.dataset.data_prefix.img_path = 'images'
    config.val_dataloader.dataset.data_root = data_root_path

    # => model
    config.model.bbox_head.num_classes = 1
    config.train_dataloader.batch_size = batch_size
    config.max_epochs = epochs

    # => training experiments
    config.work_dir = working_dir
    config.gpu_devices = [0]

    print(f"Modified config")
    print(config)

    # Load model
    runner = Runner.from_cfg(config)
    print(runner)

    # Training
    runner.train()
    print("Success")


if __name__ == '__main__':
    main()