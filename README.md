# model-api

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
```

## Information about GPU Training

If you are going to run this program on a MacOS device with Apple Silicon, you are going select the device as "mps" (Metal Performance Shaders) for GPU training acceleration.
This code repository is created for this type of devices, so the following codes in api.py file are related to that:

```
device = "mps" // if you are going to use a windows computer, change this to CUDA and install it to your computer.
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

After you run the program, it will start on port _127.0.0.1:8000_,

Then, you can go to the user interface on *http://127.0.0.1:8000/docs#*

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
