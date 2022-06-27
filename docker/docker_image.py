import argparse
import os
import sys
from omegaconf import ValidationError

from cloud.AwsConfig import AwsConfig

def parseArgs():
    parser = argparse.ArgumentParser(description="Build docker file and push to AWS ECR")
    
    parser.add_argument(
        dest ="mode",
        help ="One of either: ".format(", ".join(["build", "build+push", "push"])),
        type = str
    )
    
    parser.add_argument(
        "--imageName",
        dest = "imageName",
        help = "The name of the image you want to build/push, will be in the format: {}/imageName".format(AwsConfig.dockerUser),
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--device",
        dest = "device",
        help = "Build on either cpu, gpu or both",
        type = str,
        required=False,
        default = "both"
    )
    
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()

def buildImage(imageName, device):
    if device == "cpu" or device == "both":
        print("building cpu docker image ....")
        os.system("docker build -t {}/{} -f docker/Dockerfile.cpu .".format(AwsConfig.dockerUser, imageName+"_cpu"))
        
    if device == "gpu" or device == "both":
        print("building gpu docker image ....")
        os.system("docker build -t {}/{} -f docker/Dockerfile.gpu .".format(AwsConfig.dockerUser, imageName+"_gpu"))
    
def pushImage(imageName, cpuRepoName, gpuRepoName, device):
    
    #authenticating to aws
    print("authenticating to aws ...")
    os.system("aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 248494052625.dkr.ecr.eu-west-1.amazonaws.com")
    
    if device == "cpu" or device == "both":
        print("pushing cpu docker image ....")
        #tag image with ECR URL
        os.system("docker tag {}/{}:latest {}".format(AwsConfig.dockerUser, imageName+"_cpu", cpuRepoName))
        #push image to ECR
        os.system("docker push {}".format(cpuRepoName))
        
    if device == "gpu" or device == "both":
        print("pushing cpu docker image ....")
        #tag image with ECR URL
        os.system("docker tag {}/{}:latest {}".format(AwsConfig.dockerUser, imageName+"_gpu", gpuRepoName))
        #push image to ECR
        os.system("docker push {}".format(gpuRepoName))
    

def inputValidation(mode, imageName, tagName):
    modeValues = ["build", "build+push", "push"]
    assert mode in modeValues, ValidationError("mode must be one of: {}".format(", ".join(modeValues)))
    
    if mode=="build":
        assert imageName != None, ValidationError("please specify an imageName")
    elif mode=="build+push":
        assert imageName != None, ValidationError("please specify an imageName")
        assert tagName != None, ValidationError("please specify a ECR URL in aws_config.py")
    elif mode=="push":
        assert imageName != None, ValidationError("please specify an imageName")
        assert tagName != None, ValidationError("please specify a ECR URL in aws_config.py")
        
def main(args):
    inputValidation
    
    if args.mode == "build":
        buildImage(args.imageName, args.device)
    elif args.mode == "build+push":
        buildImage(args.imageName, args.device)
        pushImage(args.imageName, AwsConfig.cpuEcrUrl, AwsConfig.gpuEcrUrl, args.device)
    elif args.mode == "push":
        pushImage(args.imageName, AwsConfig.cpuEcrUrl, AwsConfig.gpuEcrUrl, args.device)
        
    print("complete")

if __name__ == "__main__":
    args = parseArgs()
    main(args)