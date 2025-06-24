import google.generativeai as genai
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


# Google AI Settings
genai.configure(api_key="AIzaSyCJM-AzGpg_ey8mE8V7sOJTvUCh12Ygq-0")
vertexai.init(project="text-to-image-463914", location="us-central1")
image_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")


def image_generation():
    prompt = entry.get()

    if select_box.get() == "Vertex AI":
        try:
            images = image_model.generate_images(
                prompt=prompt,
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


    elif select_box.get() == "Diffusion Model":
        pass

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
select_box = ttk.Combobox(frame, values=["Vertex AI", "Diffusion Model"], width=15, state="readonly")
select_box.current(0)
select_box.grid(row=1, column=3, padx=5, pady=5)

ttk.Button(Main_page_tab, text="Generate Image", command=image_generation).pack(pady=10)

#Showing image in other tab
Image_label = tk.Label(Image_tab)
Image_label.pack(pady=10)

root.mainloop()
