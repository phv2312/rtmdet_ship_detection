from mmengine.runner import Runner
from mmengine.config import Config

# from ship_dataset import ShipDataset


def main():
    config_path = "configs/co_dino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py"

    working_dir = "./runs"

    batch_size = 1
    epochs = 15

    # Load
    config = Config.fromfile(config_path)
    print(f"Loaded config from path: {config}...")

    # Modify dataset type and path
    # => dataset & dataloader
    # => model
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
