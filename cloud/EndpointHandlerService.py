from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer

from cloud.SevenSegmentInferenceHandler import SevenSegmentInferenceHandler
from adet.config import get_cfg


class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md
    """
    def __init__(self):
        # define config for handler
        configFile = "/opt/ml/code/AdelaiDet/configs/BAText/SevenSegment/attn_R_50.yaml"
        opts = ["MODEL.WEIGHTS", "/opt/ml/code/models/v1_pretrain_attn_R_50.pth",
                "MODEL.DEVICE", "cuda"]
        confidenceThreshold = 0.3
        
        cfg = self.setup_cfg(configFile, opts, confidenceThreshold)
        
        transformer = Transformer(default_inference_handler=SevenSegmentInferenceHandler(cfg))
        super(HandlerService, self).__init__(transformer=transformer)
        
        
    def setup_cfg(configFile, opts, confidenceThreshold):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(configFile)
        cfg.merge_from_list(opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidenceThreshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidenceThreshold
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidenceThreshold
        cfg.MODEL.MEInst.INFERENCE_TH_TEST = confidenceThreshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidenceThreshold
        cfg.freeze()
        return cfg