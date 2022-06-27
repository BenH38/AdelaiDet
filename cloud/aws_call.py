# call aws endpoint with test image
import argparse
import base64
import json
import sys
import cv2
import requests


def parseArgs():
    parser = argparse.ArgumentParser(description="Train script in container, either in AWS SageMaker or local")
    
    parser.add_argument(
        '--endpointUrl', 
        dest = "endpointUrl",
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
        "--output",
        dest = "outputMode",
        help = "The mode you wish to display the output response as. Either JSON or VIS",
        type = str,
        required = False,
        default="JSON"
    )
    
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()
  
# implements: https://linuxtut.com/en/8da21f52e379469d744b/  
def request(endpointUrl, img):
  #Convert image to sendable format and store in JSON
  _, encimg = cv2.imencode(".png ", img)
  img_str = encimg.tostring()
  img_byte = base64.b64encode(img_str).decode("utf-8")
  img_json = json.dumps({'image': img_byte}).encode('utf-8')
  
  #Send HTTP request
  #TODO clean up status code handling and move print into main
  response = requests.post(endpointUrl, data=img_json)
  print('{0} {1}'.format(response.status_code, json.loads(response.text)["message"]))
  
  return response


def main(args):
    # fetch image
    img = cv2.imread(args.input)
    
    # request/respond endpoinnt
    response = request(args.endpointUrl, img)
    
    #TODO build out visualiser


if __name__ == "__main__":
    args = parseArgs()
    main(args)