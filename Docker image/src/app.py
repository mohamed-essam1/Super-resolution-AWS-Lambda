# app.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import base64
import boto3
import json

s3_client = boto3.client('s3')

    
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=2, padding=4), nn.Sigmoid())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
    
# Load the super resolution model (replace with your own model loading logic)
def load_model():
    # Specify the S3 bucket and file path of the PyTorch file
    bucket_name = 'aws-lambda-super-resolution-trigger'
    object_key = 'generator.pth'
    
    model = GeneratorResNet()
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    state = torch.load(BytesIO(response["Body"].read()), map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model

# Super resolution function
def super_resolve(input_image, model):
    # Your super resolution logic here
    transform = transforms.Compose([
                transforms.Resize((64, 64), Image.BICUBIC),
                transforms.ToTensor(),
            ])

    image = input_image[input_image.find(",") + 1 :]
    dec = base64.b64decode(image + "===")
    image = Image.open(BytesIO(dec))
    image = image.convert("RGB")
    print('Image Decoded---------------------------------------------------')
    input_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_image)
        print('Generation Done--------------------------------------------------')

    output_image = transforms.ToPILImage()(output[0].cpu())
    output_buffer = BytesIO()
    output_image.save(output_buffer, format="JPEG")
    print('Done--------------------------------------------------')
    output_image_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
    print('Image Encoded--------------------------------------------------')

    return output_image_base64

# Lambda handler function
def lambda_handler(event, context):
    # Load the super resolution model
    model = load_model()
    print('Model Loaded--------------------------------------------------')

    # warming up the lambda
    if event.get("source") in ["aws.events", "serverless-plugin-warmup"]:
        print("Lambda is warm!")
        return {}
    
    # Get the input image from the Lambda event
    data = json.loads(event["body"])
    print("data keys :", data.keys())
    input_image_base64 = data["image"]

    # Perform super resolution
    output_image_base64 = super_resolve(input_image_base64, model)

    result = {
        'output_image': output_image_base64
    }
    # Return the output image
    return {
        "statusCode": 200,
        "body": json.dumps(result),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }