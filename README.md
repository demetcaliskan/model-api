# model-api

You can check our deployed api from this link: https://default-service-xxuoeno7nq-ew.a.run.app/docs#/

## Dependencies

First, you have to install the following dependencies: (you can install them using either pip, pip3 or conda)

```
python
uvicorn
fastapi
torch
PIL
random
diffusers
transformers
langdetect
pydantic
base64
```

## Information about GPU Training

If you are going to run this program on a MacOS device with Apple Silicon, you are going select the device as "mps" (Metal Performance Shaders) for GPU training acceleration.
This code repository is created for this type of devices, so the following codes in api.py file are related to that:

```
device = "mps" // if you are going to use a windows computer, change this to CUDA and install it to your computer.
```

In order to work with mps, you need to run the following command:

```
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

However, if you are going to deploy the API, like we did with Docker and Google Cloud Run, you need to change the device as below:

```
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Running the Program

Before you run the program, you have to create a new file called `auth_token.py`

This file will store your hugging face access key and it will be ignored via .gitignore file

This file should have the following code:

```
auth_token = "[YOUR_HUGGING_FACE_ACCESS_KEY]"
```

In order to start this program run the following command:

```
uvicorn api:app --reload
```

### Do not forget to change your device type to mps if you are runnging locally and you are using a MacOS device with Apple Silicon!

After you run the program, it will start on port _127.0.0.1:8000_,

Then, you can go to the user interface on *http://127.0.0.1:8000/docs#*

## Deploying the API

In order to deploy the API to Google Cloud Run, we followed the instructions on this link: 
https://towardsdatascience.com/how-to-deploy-and-test-your-models-using-fastapi-and-google-cloud-run-82981a44c4fe

For these steps to work, you need to install Docker to your device and authenticate with Google Cloud Run.

If you are using a MacOS device with apple silicon, you need to change the following command:

```
docker build -t default-service-fastpai:latest .
```

with this:

```
docker build --platform linux/amd64 -t default-service-fastapi:latest .
```

Also, since our api is rather large, we changed the following command:

```
 gcloud run deploy default-service \
      --image europe-west1-docker.pkg.dev/tigers-ai/ml-images/default-service-fastapi \
      --region europe-west1 \
      --port 80 \
      --memory 4Gi
```

with this:

```
gcloud run deploy default-service \
      --image europe-west1-docker.pkg.dev/tigers-ai/ml-images/default-service-fastapi \
      --region europe-west1 \
      --port 80 \
      --memory 16Gi \
      --cpu 4 \
      --max-instances=1
```

You can check our deployed api from this link: https://default-service-xxuoeno7nq-ew.a.run.app/docs#/

## Structure of the API

Right now there is one endpoint "/" which is used to generate image to image transitions.

The endpoint takes three inputs:

- prompt: The command that users are giving to the model for desired changes,
- url: The url of the image that they desire to change,
- title: Title they want to give to the generated image

The new generated image is stored to images directory, and can be accessed from:
`http://127.0.0.1:8000/images/[imagetitle.png]`

The response of this endpoint is a JSON object with the following data fields:

```
{ "image":
  {
    "src": new generated image url
    "alt": title the user have provied
  }
}
```
