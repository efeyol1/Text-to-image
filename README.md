# 🧠 AI-Based Voice/Text Prompt-to-Image Generator

This project enables users to generate rich, high-quality images from either **text** or **voice** input. It enhances the user's prompt using **Google Gemini** and generates the final image using either **Google Vertex AI Imagen 3.0** or **Hugging Face Stable Diffusion**.

---

## 📌 Features

-  Voice and 📝 Text input support
-  Prompt enhancement using Google Gemini
-  Image generation using:
  - Google Vertex AI (Imagen 3.0)
  - Hugging Face (Stable Diffusion)
-  Two interfaces:
  - GUI with `Tkinter`
  - CLI (Command Line Interface)

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/efeyol1/Text-to-image.git
cd text-to-image
pip install -r requirements.txt   

📂 Project Structure
    text-to-image/
    ├── main.py                  # GUI version
    ├── CLI_version.py           # Command-line version
    ├── requirements.txt         # Python packages
    ├── README.md                # Documentation
    ├── prompts_samples/         # Prompt & result samples
    │   └── sample1.txt
    └── images/                  # Generated image files
        └── generated_image.jpg
