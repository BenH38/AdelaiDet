# call aws endpoint with test image
import argparse
import base64
import colorsys
import json
import sys
import cv2
import numpy as np
import requests
from sagemaker.predictor import Predictor
import matplotlib.colors as mplc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches

from cloud.AwsConfig import AwsConfig
import boto3
#Self contained visualiser for shipping
class Visualiser():
    
    def __init__(self, img):
        self.vocSize = 12
        self.CTLABELS = [48,49,50,51,52,53,54,55,56,57] #unicode representation of 0-9
        
        self.img = img
        self.imWidth = img.shape[0]
        self.imHeight = img.shape[1]
        
        plt.imshow(img)
        
        # too small texts are useless, therefore clamp to 9
        self.defaultFontSize = max(
            np.sqrt(self.imHeight * self.imWidth) // 90, 10 // 1
        )
        
    def show(self):
        plt.show()
    
    def drawPredictions(self, predictions):
        predictions = predictions["output"]
        beziers = np.array(predictions["beziers"])
        scores = predictions["scores"]
        recs = predictions["recs"]

        self.overlayInstances(beziers, recs, scores)
        
        
    def overlayInstances(self, beziers, recs, scores):
        color = (0.1, 0.2, 0.5)

        for bezier, rec, score in zip(beziers, recs, scores):
            polygon = self.bezierToPoly(bezier)
            self.drawPolygon(polygon, color, alpha=0.5)

            # draw text in the top left corner
            text = self.decodeRecognition(rec)
            text = "{:.3f}: {}".format(score, text)
            lighter_color = self.changeColorBrightness(color, brightness_factor=0.7)
            text_pos = polygon[0]
            horiz_align = "left"
            font_size = self.defaultFontSize

            self.drawText(
                text,
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size
            )
        
    def bezierToPoly(self, bezier):
        # bezier to polygon
        u = np.linspace(0, 1, 20)
        bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
        points = np.outer((1 - u) ** 3, bezier[:, 0]) \
                + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
                + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
                + np.outer(u ** 3, bezier[:, 3])
        points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)

        return points

    def drawPolygon(self, segment, color, edge_color=None, alpha=0.5):
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self.defaultFontSize // 15 * 1, 1),
        )
        plt.gca().add_patch(polygon)

    def decodeRecognition(self, rec):
        s = ''
        for c in rec:
            c = int(c)
            if c < self.vocSize - 1:
                if self.vocSize == 96:
                    s += self.CTLABELS[c]
                else:
                    s += str(chr(self.CTLABELS[c]))
            elif c == self.vocSize -1:
                #s += u'口'
                pass
        return s

    def changeColorBrightness(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def drawText(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW
        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self.defaultFontSize

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        
        x, y = position
        
        plt.gca().text(
                x,
                y,
                text,
                size=font_size * 1,
                family="sans-serif",
                bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
        )

def parseArgs():
    parser = argparse.ArgumentParser(description="Train script in container, either in AWS SageMaker or local")
    
    parser.add_argument(
        '--method', 
        dest = "requestMethod",
        help = "Request to the endpoint using AWS API Gateway, AWS SDK, or to a local server. [external, sdk, local]",
        type=str, 
        required=True, 
    )
    
    parser.add_argument(
        '--endpoint', 
        dest = "endpointName",
        help = "URL of the sagemaker endpoint to be called",
        type=str, 
        required=True, 
    )
    
    parser.add_argument(
        "--input",
        dest = "input",
        help = "The directory of the input image. Server currently only supports one image",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--vis",
        dest = "vis",
        help = "Specify if you wish to visualise the predictions on the original image",
        default=False,
        action="store_true"
    )
    
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()

# implements: https://linuxtut.com/en/8da21f52e379469d744b/  
def makePayload(img):
    #Convert image to sendable format and store in JSON
    _, encimg = cv2.imencode(".png ", img)
    img_str = encimg.tostring()
    img_byte = base64.b64encode(img_str).decode("utf-8")
    payload = json.dumps({'image': img_byte})#.encode('utf-8')
    
    return payload
    
def sdkRequest(endpointName, img):
    payload = makePayload(img)
    """
    #Send HTTP request
    #TODO clean up status code handling and move print into main
    predictor = Predictor(endpointName, content_type='application/json')
    response = predictor.predict(data=payload, )
    """
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(EndpointName=endpointName, Body=payload, ContentType="application/json")
    
    result = json.loads(response['Body'].read().decode())
    
    return result

def externalRequest(img):
    payload = makePayload(img)
    
    response = requests.post(AwsConfig.apiEnpointUrl, json=payload)
    
    print("RESPONSE CODE: " + str(response.status_code))
    
    return response.json()

def localRequest(img):
    payload = makePayload(img)
    
    response = requests.post("http://localhost:{}/invocations".format(AwsConfig.sagemakerInferencePort), json=payload)
    
    print("RESPONSE CODE: " + str(response.status_code))
    
    return response.json()

def main(args):
    # Read image into memory
    img = cv2.imread(args.input)

    if args.requestMethod == "local":
        response = localRequest(img)
    elif args.requestMethod == "sdk":
        response = sdkRequest(args.endpointName, img)
    elif args.requestMethod == "external":
        response = externalRequest(img)
    else:
        raise AssertionError("ERROR: Invalid request argument. [external, sdk, local]")
    
    print(response)
    
    if args.vis:
        visualiser = Visualiser(img)
        visualiser.drawPredictions(response)
        visualiser.show()
    


if __name__ == "__main__":
    args = parseArgs()
    main(args)