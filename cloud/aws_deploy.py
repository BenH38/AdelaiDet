import argparse
import os
import sys
from sagemaker.estimator import Estimator

from cloud.AwsConfig import AwsConfig
from docker.docker_image import buildImage, pushImage

def parseArgs():
    parser = argparse.ArgumentParser(description="Train script in container, either in AWS SageMaker or local")
    
    parser.add_argument(
        dest ="mode",
        help ="One of either: ".format(", ".join(["local", "aws"])),
        type = str,
    )
    
    parser.add_argument(
        '--build', 
        help = "If specified we will also rebuild the image. If mode is set to AWS, the new image will be pushed to AWS ECR",
        nargs='?', 
        default=False, 
        const=True
    )
    
    parser.add_argument(
        "--imageName",
        dest = "imageName",
        help = "The name of the image you want to build/push, will be in the format: {}/imageName".format(AwsConfig.dockerUser),
        type = str,
        required = False,
    )
    
    parser.add_argument(
        "--instance",
        dest ="instanceType",
        help ="The instance type you want to deploy to in AWS, please check Sagemaker pricing for details",
        type = str,
        required = False,
    )

    parser.add_argument(
        "--project",
        dest = "projectName",
        help = "The base name of the project",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--job",
        dest = "jobName",
        help = "The name given to this specific job/deployment, must be unique!",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--input",
        dest = "inputDir",
        help = "The directory of the input data. For local: specify the complete file dir, for AWS: specify the internal bucket path",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--output",
        dest = "outputDir",
        help = "The directory where the output model shall be saved. Only required in local mode, for AWS, model will be saved in output bucket specified in aws_config.py",
        type = str,
        required = False,
    )
    
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()
    
def train(instanceType, image, projectName, jobName, inputDir, outputDir = None):
    
    estimator = Estimator(image_uri = image, 
                      role = AwsConfig.roleName, 
                      instance_count=1, 
                      instance_type = instanceType, 
                      output_path = outputDir, 
                      base_job_name = projectName)
    
    print("Sagemaker: Start fitting")
    estimator.fit(inputs=inputDir, job_name=jobName)
    
def main(args):
    
    if args.mode == "local":
        
        if args.build:
            buildImage(args.imageName, "cpu")
            
        outputLocation = "file://{}".format(args.outputDir)
        inputDataLocation = "file://{}".format(args.inputDir)
        
        image = "{}/{}".format(AwsConfig.dockerUser, args.imageName)
            
        train("local", image, args.projectName, args.jobName, inputDataLocation, outputLocation)
        
    elif args.mode == "aws":
        
        #check if we're using a gpu instance
        device = "gpu" if args.instanceType in AwsConfig.availableGpuInstances else "cpu"
        
        if args.build:
            buildImage(args.imageName, device)
            pushImage(args.imageName, AwsConfig.cpuEcrUrl, AwsConfig.gpuEcrUrl, device)
            
        outputLocation = "s3://{}/{}".format(AwsConfig.outputBucket, args.projectName)
        inputDataLocation = "s3://{}/{}".format(AwsConfig.inputBucket, args.inputDir)
            
        image = AwsConfig.gpuEcrUrl if device=="gpu" else AwsConfig.cpuEcrUrl
            
        train(args.instanceType, image, args.projectName, args.jobName, inputDataLocation, outputLocation)
        
        
    print("complete")

if __name__ == "__main__":
    args = parseArgs()
    main(args)