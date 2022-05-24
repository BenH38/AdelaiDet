from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors
from detectron2.engine.defaults import DefaultPredictor
import json
import base64
import numpy as np
import cv2

class SevenSegmentInferenceHandler():
    
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def default_model_fn(self, model_dir):
        """Loads a model. For PyTorch, a default function to load a model cannot be provided.
        Users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.

        Returns: A PyTorch model.
        """
        
        predictor = DefaultPredictor(self.cfg)
        

    def default_input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type

        Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
        """
        
        #Data conversion process
        #Implements: https://linuxtut.com/en/8da21f52e379469d744b/
        
        data_json = json.loads(input_data)
        image = data_json['image']
        image_dec = base64.b64decode(image)
        data_np = np.fromstring(image_dec, dtype='uint8')
        #decimg = cv2.imdecode(data_np, 1) #don't need cv2 format
        
        return data_np

    def default_predict_fn(self, data, model):
        """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn

        Returns: a prediction
        """
        
        predictions = model(data)
        
        return predictions

    def default_output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """
        return encoder.encode(prediction, accept)