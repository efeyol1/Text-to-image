import google.generativeai as genai
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
import speech_recognition as sr


# Google AI Settings
genai.configure(api_key="AIzaSyCJM-AzGpg_ey8mE8V7sOJTvUCh12Ygq-0")
vertexai.init(project="text-to-image-463914", location="us-central1")
image_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
gen_model = genai.GenerativeModel('gemini-2.0-flash-lite')

gemini_instruction = (
  "Your mission for the project is being a Prompt analyzer"
  "Analyze the first prompt and if you think it is not detailed ,"
  "You should make the prompts richer and detailed so that it can be used in image generation  "
  "If it is complicated as it is just add details in it (trying not to change the topic of it )"
  "Keep it under 100 words so that Project can run faster"
)

def image_generation(model,typeofinput):


    if typeofinput == "text" :
        prompt = input("Enter your prompt: ")

    elif typeofinput == "voice":
        r = sr.Recognizer()

        with sr.Microphone() as source:
            print("Say something!")
            # It listens for a second to calibrate noise levels.
            r.adjust_for_ambient_noise(source, duration=1)


            audio = r.listen(source, timeout=5)
            print("Processing your speech...")

            # --- Using Google Web Speech API (Online) ---
            # This is free for basic use, but requires internet.

            try:
                text = r.recognize_google(audio)
                prompt = text
                print(f"Google Web Speech API thinks you said: \"{text}\"")
            except sr.UnknownValueError:
                print("Google Web Speech API could not understand the audio")
                return
            except sr.RequestError as e:
                print(f"Could not request results from Google Web Speech API; {e}")
                return
            except audio is None:
                print("No audio input detected.")
                return


    genprompt = f"Create a detailed and organized prompt using: {prompt}"
    response = gen_model.generate_content(
            f"{gemini_instruction}\n\nUser's original prompt: '{prompt}'\n\nEnhanced prompt:")

    enhanced_prompt = response.text.strip()
    print(f"Original Prompt: '{prompt}'")
    print(f"\nEnhanced Prompt from Gemini:\n'{enhanced_prompt}'")

    if model == "Vertex AI":
        try:
            images = image_model.generate_images(
                prompt=enhanced_prompt,
                number_of_images=1,
            )

            images[0].save("generated_image.jpg")

            img = Image.open("generated_image.jpg")
            img.show()
        except Exception as e:
            print(f"Image could not be generated: {str(e)}")

    elif model == "Hugging Face":
        model_id = "runwayml/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(model_id)

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            print("CUDA not available, running on CPU. This will be very slow.")
            pipe = pipe.to("cpu")

        print(f"Generating image for prompt: '{enhanced_prompt}'...")

        try:
            image = pipe(prompt=enhanced_prompt).images[0]
            image.save("generated_image.jpg")
            image.show()
        except Exception as e:
            print(f"Image generation failed: {e}")



if __name__ == '__main__':
    model = input("select a model (Vertex AI or Hugging Face): ")
    typeofinput = input("select a input type (text or voice): ")
    image_generation(model,typeofinput)