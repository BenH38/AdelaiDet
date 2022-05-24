import sys
import os

# UNSAFE
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import argparse
from sagemaker_inference import model_server
from adet.config.config import get_cfg

from cloud.EndpointHandlerService import HandlerService
from cloud.TrainHandlerService import TrainHandlerService



def parseArgs():
    parser = argparse.ArgumentParser(description="Train script in container, either in AWS SageMaker or local")
    
    parser.add_argument(
        "mode",
        help ="One of either: ".format(", ".join(["serve", "train"])),
        type = str,
    )
    
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()

def main(args):
    
    if args.mode == "serve":
        model_server.start_model_server(handler_service=HandlerService)
        
    elif args.mode =="train":
        TrainHandlerService()


if __name__=="__main__":
    args = parseArgs()
    main(args)