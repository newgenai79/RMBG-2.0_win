import os
import gradio as gr
from gradio_imageslider import ImageSlider
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import glob

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-2.0", trust_remote_code=True
)
birefnet.to("cuda")
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    image = process(im)    
    image_path = os.path.join(output_folder, "no_bg_image.png")
    image.save(image_path)
    return (image, origin), image_path

def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

def batch_process(input_dir, output_dir):
    # Validate input and output directories
    if not os.path.isdir(input_dir):
        return f"Error: Input directory {input_dir} does not exist."
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files (supporting common image formats)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)) + 
                           glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        return "No image files found in the input directory."
    
    # Process each image
    processed_images = []
    processed_count = 0
    failed_images = []

    for image_path in image_files:
        try:
            # Load the image
            im = load_img(image_path, output_type="pil")
            im = im.convert("RGB")
            
            # Process the image
            processed_image = process(im)
            
            # Create output filename preserving original filename, but with .png extension
            relative_path = os.path.relpath(image_path, input_dir)
            filename_without_ext = os.path.splitext(relative_path)[0]
            output_path = os.path.join(output_dir, f"{filename_without_ext}.png")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the processed image as PNG
            processed_image.save(output_path)
            
            processed_images.append(output_path)
            processed_count += 1
        
        except Exception as e:
            failed_images.append(f"{image_path}: {str(e)}")
            print(f"Error processing {image_path}: {e}")
    
    # Prepare result message
    result_message = f"Processed {processed_count} images successfully.\n"
    result_message += f"Output saved in {output_dir}\n"
    
    if failed_images:
        result_message += "Failed to process the following images:\n"
        result_message += "\n".join(failed_images)
    
    return result_message

slider1 = ImageSlider(label="RMBG-2.0", type="pil")
slider2 = ImageSlider(label="RMBG-2.0", type="pil")
image = gr.Image(label="Upload an image")
image2 = gr.Image(label="Upload an image", type="filepath")
text = gr.Textbox(label="Paste an image URL")
png_file = gr.File(label="output png file")

url = "http://farm9.staticflickr.com/8488/8228323072_76eeddfea3_z.jpg"

tab1 = gr.Interface(
    fn, inputs=image, outputs=[slider1, gr.File(label="output png file")], api_name="image"
)

tab2 = gr.Interface(fn, inputs=text, outputs=[slider2, gr.File(label="output png file")], examples=[url], api_name="text")

# New batch processing tab
tab3 = gr.Interface(
    batch_process, 
    inputs=[
        gr.Textbox(label="Input Directory Path"),
        gr.Textbox(label="Output Directory Path")
    ], 
    outputs=gr.Textbox(label="Processing Result"),
    title="Batch Background Removal",
    api_name="batch"
)

demo = gr.TabbedInterface(
    [tab1, tab2, tab3], 
    ["input image", "input url", "batch processing"], 
    title="RMBG-2.0 for background removal"
)

if __name__ == "__main__":
    demo.launch(show_error=True)