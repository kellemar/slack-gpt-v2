import os
# Stablity AI
import warnings
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

from PIL import Image
import io
import traceback

stability_api = client.StabilityInference(
  key=os.environ['STABILITY_KEY'],  # API Key reference.
  verbose=True,  # Print debug messages.
  engine=
  "stable-diffusion-xl-beta-v2-2-2",  # Set the engine to use for generation.
  # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
  # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)


def get_image_from_stability(image_prompt):
  result = {}
  result['output'] = ""
  result['image_generated'] = ""
  try:
    answers = stability_api.generate(
      prompt=image_prompt,
      steps=
      30,  # Amount of inference steps performed on image generation. Defaults to 30.
      cfg_scale=
      7.0,  # Influences how strongly your generation is guided to match your prompt.
      # Setting this value higher increases the strength in which it tries to match your prompt.
      # Defaults to 7.0 if not specified.
      width=512,  # Generation width, defaults to 512 if not included.
      height=512,  # Generation height, defaults to 512 if not included.
      samples=1,  # Number of images to generate, defaults to 1 if not included.
      sampler=generation.
      SAMPLER_K_DPMPP_2M  # Choose which sampler we want to denoise our generation with.
      # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
      # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
    )
    for resp in answers:
      for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
          warnings.warn(
            "Your request activated the API's safety filters and could not be processed."
            "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
          img = Image.open(io.BytesIO(artifact.binary))
          img_filename = "./images/" + str(artifact.seed) + ".png"
          img.save(img_filename)
          #result['output'] = "Here is the image you requested. It was generated from the prompt:\n" + image_prompt
          #result['image_generated'] = base64.b64encode(
          #artifact.binary).decode('utf-8')
          result = "Here is the image you requested. It was generated from the prompt:\n" + image_prompt
          result = result + "File:" + img_filename
          return result
  except Exception as e:
    traceback.print_exc()
    result[
      'output'] = "There is an issue with creating your image. Please try another prompt."
    return result
