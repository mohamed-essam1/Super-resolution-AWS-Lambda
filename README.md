# Super Resolution using Generative Adversarial Networks (GANs) with PyTorch

![Super Resolution](https://img.shields.io/badge/super-resolution-brightgreen.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9.1-orange.svg)

---

## Overview

This project implements a super-resolution model using Generative Adversarial Networks (GANs) with PyTorch. The goal is to enhance the quality of images by upscaling them from a resolution of 64x64 to 128x128.

## How it Works

The super-resolution model is trained using a GAN architecture, consisting of a generator and a discriminator. The generator aims to generate high-resolution images from low-resolution inputs, while the discriminator learns to distinguish between real high-resolution images and generated ones.

## Getting Started

### Prerequisites

- Python 3.9 or later
- PyTorch 1.9.1
- AWS Lambda account for deployment


## Deployment Steps

### 1. Configure your AWS credentials

Ensure you have AWS CLI installed and configured with your AWS credentials.

```bash
aws configure
```

### 2. Create ECR and login to this repo

Create an Amazon Elastic Container Registry (ECR) repository to store your Docker image.

```bash
aws ecr create-repository --repository-name super-resolution-gan
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
```

### 3. Build Docker Image

Build the Docker image containing your super-resolution model code and dependencies.

```bash
docker build -t super-resolution-gan .
```

### 4. Tag Docker Image

Tag the Docker image with the ECR repository information.

```bash
docker tag super-resolution-gan:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/super-resolution-gan:latest
```

### 5. Push Docker Image to ECR

Push the Docker image to the ECR repository.

```bash
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/super-resolution-gan:latest
```

### 6. Create AWS Lambda and connect it with S3 bucket

Create an AWS Lambda function with the following settings:

- **Function Name:** super-resolution-lambda
- **Runtime:** Python 3.9 or later
- **Handler:** app.lambda_handler
- **Role:** Create a role with Lambda and S3 access permissions
- **Existing Role:** Choose the role you created
- **Code Entry Type:** Container Image
- **Image URI:** `<your-account-id>.dkr.ecr.<your-region>.amazonaws.com/super-resolution-gan:latest`

### 7. Generate API Gateway

Create an API Gateway to expose your Lambda function as an HTTP endpoint:

- In the Lambda function configuration, add an API Gateway trigger.
- Create a new HTTP API in API Gateway.
- Deploy the API to a new or existing stage.

Now, your super-resolution model is deployed on AWS Lambda and accessible through the API Gateway endpoint.

Remember to secure your API Gateway using authentication and authorization mechanisms if required for your application.

## Usage

### API Endpoint

Make a POST request to the API endpoint with a base64-encoded low-resolution image:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image": "base64_encoded_image"}' https://your-api-endpoint.execute-api.region.amazonaws.com/dev/super-resolution
```

Replace `base64_encoded_image` with the base64 encoding of the low-resolution image, and `your-api-endpoint` and `region` with the corresponding values from the deployment.

## Integration with Mobile App or Website

Integrate the API endpoint into your mobile application or website by making HTTP requests to the deployed endpoint. Use the provided example to guide the integration process.
---
