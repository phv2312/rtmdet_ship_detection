from mmengine.registry import DATASETS
from mmrotate.datasets.dota import DOTADataset


@DATASETS.register_module()
class ShipDataset(DOTADataset):
    """SAR ship dataset for detection."""
    METAINFO = {
        'classes':
            ('ship',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42)]
    }
