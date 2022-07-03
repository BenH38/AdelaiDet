from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors
from detectron2.engine.defaults import DefaultPredictor
import json
import base64
import numpy as np
import cv2
import flask
import logging
import os

from adet.config.config import get_cfg
from cloud.AwsConfig import AwsConfig

class InferenceHandlerService():
    
    def __init__(self):
        if os.environ.get("IS_THIS_DOCKER_ENVIRONMENT") == "yes": #TODO get external config file specifying this info
            # define config for handler
            if os.environ.get("CPU_OR_GPU_RUNTIME") == "cpu":
                device = "cpu"
            elif os.environ.get("CPU_OR_GPU_RUNTIME") == "gpu":
                device = "cuda"
        
            configFile = "/opt/ml/code/AdelaiDet/configs/BAText/SevenSegment/attn_R_50.yaml"
            opts = ["MODEL.WEIGHTS", os.path.join(AwsConfig.savedModelDir, AwsConfig.savedModelName),
                    "MODEL.DEVICE", device]
        else:
            configFile = "configs/BAText/SevenSegment/attn_R_50.yaml"
            opts = ["MODEL.WEIGHTS", "models/aws_test_model_4.pth",
                    "MODEL.DEVICE", "cpu"]
        
        confidenceThreshold = 0.3 #TODO get from external config file
        
        cfg = self.setup_cfg(configFile, opts, confidenceThreshold)
        
        self.predictor = DefaultPredictor(cfg)
        
        self.server()
        
    def setup_cfg(self, configFile, opts, confidenceThreshold):
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
    
    def server(self):
        app = flask.Flask(__name__)

        @app.route("/ping", methods=["GET"])
        def ping():
            # Check if the classifier was loaded correctly TODO
            try:
                #self.predictor
                status = 200
                logging.info("Status : 200")
            except:
                status = 400
            return flask.Response(response= json.dumps(""), status=status, mimetype="application/json" )
        
        @app.route("/invocations", methods=["POST"])
        def predict():
            # Get input JSON data and convert it to a DF
            inputJson = flask.request.get_json()
            print(inputJson)
            #dataJson = json.loads(inputJson)
            image = inputJson['image']
            imageDec = base64.b64decode(image)
            npData = np.fromstring(imageDec, dtype='uint8')
            decimg = cv2.imdecode(npData, 1)
            
            predictions = self.predictor(decimg)
            
            output = self.instancesToOutput(predictions)

            # Transform predictions to JSON
            result = {
                "output": output
                }

            resultjson = json.dumps(result)
            return flask.Response(response=resultjson, status=200, mimetype="application/json", content_type="application/json")
        
        app.run(host="0.0.0.0", port=AwsConfig.sagemakerInferencePort, debug=True)
        
    def instancesToOutput(self, predictions):
        instances = predictions["instances"]
        
        numInstances = len(instances)
        imageSize = instances.image_size

        bboxes = instances.pred_boxes.tensor.tolist()
        scores = instances.scores.tolist()
        predClasses = instances.pred_classes.tolist()
        recs = instances.recs.tolist()
        beziers = instances.beziers.tolist()
        
        output = {
            "image_size": imageSize,
            "num_instances": numInstances,
            "bboxes": bboxes,
            "scores": scores,
            "pred_classes": predClasses,
            "recs": recs,
            "beziers": beziers
        }
        
        return output