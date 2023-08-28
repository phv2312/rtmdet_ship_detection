import cv2
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmrotate.structures.bbox import QuadriBoxes, RotatedBoxes


def draw_predict(image: np.ndarray, bboxes):
    if isinstance(bboxes, Tensor):
        if bboxes.size(-1) == 5:
            bboxes = RotatedBoxes(bboxes)
        elif bboxes.size(-1) == 8:
            bboxes = QuadriBoxes(bboxes)
        else:
            raise TypeError(
                'Require the shape of `bboxes` to be (n, 5) '
                'or (n, 8), but get `bboxes` with shape being '
                f'{bboxes.shape}.')

    bboxes = bboxes.cpu()
    polygons = bboxes.convert_to('qbox').tensor
    polygons = polygons.reshape(-1, 4, 2)
    polygons = [p.cpu().numpy().astype(int) for p in polygons]

    visual = image.copy()
    for polygon in polygons:
        cv2.drawContours(visual, [polygon[:, None, :]], -1, color=(255, 0, 0), thickness=2)
    return visual


def predict(checkpoint_path: str, config_path: str, image_path: str):
    config = Config.fromfile(config_path)
    config.model.bbox_head.num_classes = 1

    model = init_detector(config, checkpoint_path, device='cuda:0')

    results = inference_detector(model, image_path)
    print(results)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    visual = draw_predict(image, results.pred_instances.bboxes)

    plt.imshow(visual)
    plt.show()


if __name__ == '__main__':
    checkpoint_path_in = "<your_weight>"
    config_path_in = "./configs/rotated_rtmdet/rotated_rtmdet_l-100e-aug-ship.py"
    image_path_in = "./data/ssdd_tiny/images/000009.png"

    predict(checkpoint_path_in, config_path_in, image_path_in)
