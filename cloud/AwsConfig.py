from matplotlib.style import available


class AwsConfig():
    
    roleName = "SageMaker"
    
    cpuEcrUrl = "248494052625.dkr.ecr.eu-west-1.amazonaws.com/sevensegment_cpu"
    gpuEcrUrl = "248494052625.dkr.ecr.eu-west-1.amazonaws.com/sevensegment_gpu"
    
    inputBucket = "blueye-training-data"
    outputBucket = "blueye-ml-models"
    
    awsLocation = "eu-west-1"
    
    dockerUser = "benh38"
    
    availableGpuInstances = ["ml.p4d.24xlarge",
                                "ml.p3.2xlarge",
                                "ml.p3.8xlarge",
                                "ml.p3.16xlarge",
                                "ml.p3dn.24xlarge",
                                "ml.p2.xlarge",
                                "ml.p2.8xlarge",
                                "ml.p2.16xlarge",
                                "ml.g4dn.xlarge",
                                "ml.g4dn.2xlarge",
                                "ml.g4dn.4xlarge",
                                "ml.g4dn.8xlarge",
                                "ml.g4dn.12xlarge",
                                "ml.g4dn.16xlarge"]
    