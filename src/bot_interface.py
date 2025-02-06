from PIL import Image
import os

# Load the model and processor
model_name = "Salesforce/instructblip-flan-t5-xl"

def prompt_InstructBlip(image_path, prompt, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    output_ids = model.generate(**inputs)
    return processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

