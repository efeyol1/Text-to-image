import google.generativeai as genai
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
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


def image_generation(input1=None):
    if input1 is None:
        prompt = entry.get()
    else:
        prompt = input1


    genprompt = f"Create a detailed and organized prompt using: {prompt}"
    response = gen_model.generate_content(
            f"{gemini_instruction}\n\nUser's original prompt: '{prompt}'\n\nEnhanced prompt:")

    enhanced_prompt = response.text.strip()
    print(f"Original Prompt: '{prompt}'")
    print(f"\nEnhanced Prompt from Gemini:\n'{enhanced_prompt}'")


    if select_box.get() == "Vertex AI":
        try:
            images = image_model.generate_images(
                prompt=enhanced_prompt,
                number_of_images=1,
            )

            images[0].save("generated_image.jpg")

            img = Image.open("generated_image.jpg")
            tk_image = ImageTk.PhotoImage(img)

            # Show it on UI
            Image_label.config(image=tk_image)
            Image_label.image = tk_image
        except Exception as e:
            messagebox.showerror("Hata", f"Görsel oluşturulamadı: {str(e)}")


    elif select_box.get() == "Hugging Face":
        model_id = "runwayml/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(model_id)

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            print("CUDA not available, running on CPU. This will be very slow.")
            pipe = pipe.to("cpu")


        print(f"Generating image for prompt: '{enhanced_prompt}'...")

        image = pipe(prompt).images[0]  # Simplest text-to-image

        #Display the image
        image.save("generated_image.jpg")

        Image_label.config(image=image)
        Image_label.image = image


def transcribe_microphone_input():
    r = sr.Recognizer()  # Initialize the recognizer

    with sr.Microphone() as source:
        print("Say something!")
        # It listens for a second to calibrate noise levels.
        r.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = r.listen(source, timeout=5)  # Listen for up to 5 seconds of speech
            print("Processing your speech...")

            # --- Using Google Web Speech API (Online) ---
            # This is free for basic use, but requires internet.
            text = r.recognize_google(audio)
            print(f"Google Web Speech API thinks you said: \"{text}\"")
            return image_generation(text)

        except sr.WaitTimeoutError:
            print("No speech detected within the timeout period.")
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from the speech recognition service; {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return None


# GUI setup
root = tk.Tk()
root.title("Intelligent Text-to-Image Generator")
root.geometry("1000x750")

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

Main_page_tab = ttk.Frame(notebook)
notebook.add(Main_page_tab, text='Main Page')

Image_tab = ttk.Frame(notebook)
notebook.add(Image_tab, text="Image")

tk.Label(Main_page_tab, text="Write a text").pack(pady=(10, 5))
entry = ttk.Entry(Main_page_tab, width=45)
entry.pack()

frame = ttk.Frame(Main_page_tab)
frame.pack(pady=10)

ttk.Label(frame, text="Choose a model to use:").grid(row=1, column=2, padx=5, pady=5)
select_box = ttk.Combobox(frame, values=["Vertex AI", "Hugging Face"], width=15, state="readonly")
select_box.current(0)
select_box.grid(row=1, column=3, padx=5, pady=5)

ttk.Button(Main_page_tab, text="Generate Image", command= lambda: image_generation()).pack(pady=10)
ttk.Button(Main_page_tab, text="Voice input", command=transcribe_microphone_input).pack(pady=10)

#Showing image in other tab
Image_label = tk.Label(Image_tab)
Image_label.pack(pady=10)

root.mainloop()
