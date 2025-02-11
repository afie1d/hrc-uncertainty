from PIL import Image
import torch

def prompt_InstructBlip(image_paths, prompts, processor, model):
    images = [Image.open(path).convert("RGB") for path in image_paths]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(images=images, text=prompts, return_tensors="pt")
    inputs.to(device)

    outputs = model.generate(**inputs)
    return processor.batch_decode(outputs, skip_special_tokens=True)

