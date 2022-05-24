from adet.config.config import get_cfg
from detectron2.engine import default_argument_parser, launch
import os
from tool.train_net import main as Train


class TrainHandlerService():
    
    def __init__(self) -> None:
        
        # check if we're in a docker environment, and whether to cpu or gpu
        if os.environ.get("IS_THIS_DOCKER_ENVIRONMENT") == "yes":
            if os.environ.get("CPU_OR_GPU_RUNTIME") == "cpu":
                configFile = "/opt/ml/code/AdelaiDet/configs/BAText/SevenSegment/aws_cpu_attn_R_50.yaml"
            elif os.environ.get("CPU_OR_GPU_RUNTIME") == "gpu":
                configFile = "/opt/ml/code/AdelaiDet/configs/BAText/SevenSegment/aws_gpu_attn_R_50.yaml"
        else: 
            configFile = "configs/BAText/SevenSegment/attn_R_50.yaml"
        
        args = default_argument_parser().parse_args(
            ["--config-file", configFile]
        )
        
        launch(
            Train,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
