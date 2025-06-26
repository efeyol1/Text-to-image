import os
import google.generativeai as genai
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# ==== CONFIG ====
API_KEY = "AIzaSyCJM-AzGpg_ey8mE8V7sOJTvUCh12Ygq-0"
PROJECT_ID = "text-to-image-463914"
MODEL = "Hugging Face"
ORIGINAL_PROMPT = input("Enter original prompt: ").strip()
sample_index = len(os.listdir("Sample_Prompts")) + 1
output_image_path = f"images/generated_image_{sample_index}.jpg"
sample_txt_path = f"Sample_Prompts/sample{sample_index}.txt"
# ================

# Init Gemini & Vertex
genai.configure(api_key=API_KEY)
vertexai.init(project=PROJECT_ID, location="us-central1")
gen_model = genai.GenerativeModel('gemini-2.0-flash-lite')
image_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

# Gemini prompt enhancement
instruction = (
    "Your mission for the project is being a Prompt analyzer. "
    "Analyze the first prompt and if you think it is not detailed, "
    "you should make the prompts richer and detailed so that it can be used in image generation. "
    "If it is complicated as it is, just add details in it (trying not to change the topic of it). "
    "Keep it under 75 words so that the project can run faster."
)
full_prompt = f"{instruction}\n\nUser's original prompt: '{ORIGINAL_PROMPT}'\n\nEnhanced prompt:"
enhanced = gen_model.generate_content(full_prompt).text.strip()

# Image generation
if MODEL == "Vertex AI":
    images = image_model.generate_images(prompt=enhanced, number_of_images=1)
    image_bytes = images[0]._image_bytes
    with open(output_image_path, "wb") as f:
        f.write(image_bytes)

elif MODEL == "Hugging Face":
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    image = pipe(prompt=enhanced).images[0]
    image.save(output_image_path)

# Sample file creation
with open(sample_txt_path, "w", encoding="utf-8") as f:
    f.write(f"Original Prompt:\n{ORIGINAL_PROMPT}\n\n")
    f.write(f"Enhanced Prompt (Gemini):\n{enhanced}\n\n")
    f.write(f"Model Used:\n{MODEL}\n\n")
    f.write(f"Generated Image Path:\n{output_image_path}\n")

print(f"✅ Sample saved to {sample_txt_path}")
print(f"🖼️ Image saved to {output_image_path}")
