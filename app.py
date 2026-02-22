# app.py
import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# VGG19 Normalization (ImageNet stats)
# -----------------------------
vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)

# -----------------------------
# Image Loading & Processing
# -----------------------------
def load_image(img_file, max_size=256):
    image = Image.open(img_file).convert("RGB")
    size = min(max_size, max(image.size))
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def tensor_to_image(tensor):
    tensor = tensor.cpu().clone().detach().squeeze(0)
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor)

# -----------------------------
# Gram Matrix
# -----------------------------
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)

# -----------------------------
# Load VGG19
# -----------------------------
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

content_layers = ['21']
style_layers = ['0', '5', '10', '19', '28']

# Style layer weights (deeper layers = more abstract patterns)
style_weights = {
    '0': 1.0,
    '5': 0.8,
    '10': 0.5,
    '19': 0.3,
    '28': 0.1
}

# -----------------------------
# Feature Extraction (with normalization)
# -----------------------------
def get_features(image, model, layers):
    x = (image - vgg_mean) / vgg_std
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# -----------------------------
# Neural Style Transfer (Adam optimizer)
# -----------------------------
def neural_style_transfer(content_file, style_file, steps=300, alpha=1, beta=1e6):
    content = load_image(content_file, max_size=256)
    style = load_image(style_file, max_size=256)
    target = content.clone().requires_grad_(True).to(device)

    content_features = get_features(content, vgg, content_layers)
    style_features = get_features(style, vgg, style_layers)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    optimizer = optim.Adam([target], lr=0.003)

    for step in range(1, steps + 1):
        target_features = get_features(target, vgg, content_layers + style_layers)

        content_loss = torch.mean(
            (target_features[content_layers[0]] - content_features[content_layers[0]]) ** 2
        )

        style_loss = 0
        for layer in style_layers:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

        total_loss = alpha * content_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Keep pixel values valid
        target.data.clamp_(0, 1)

        if step % 50 == 0:
            print(f"Step [{step}/{steps}], Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}, Total: {total_loss.item():.2f}")

    print("Done!")
    return tensor_to_image(target)

# -----------------------------
# Gradio Frontend
# -----------------------------
def run_nst(content_img, style_img, steps=300, alpha=1, beta=1e6):
    result = neural_style_transfer(content_img, style_img, steps=int(steps), alpha=alpha, beta=beta)
    return result

css = """
.gradio-container {
    background: linear-gradient(135deg, #0a192f, #112240, #1a365d) !important;
    min-height: 100vh;
}
.main-title {
    text-align: center;
    color: #e2e8f0;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #f6ad55;
    font-size: 1.1rem;
    margin-top: 4px;
    margin-bottom: 20px;
}
.card-section {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(246,173,85,0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}
footer { display: none !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="amber",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter")
)) as demo:

    gr.HTML("<h1 class='main-title'>Neural Style Transfer</h1>")
    gr.HTML("<p class='subtitle'>Transform your photos into artwork using AI</p>")

    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="card-section"):
            content_img = gr.Image(type="filepath", label="Content Image", height=300)
        with gr.Column(elem_classes="card-section"):
            style_img = gr.Image(type="filepath", label="Style Image", height=300)

    with gr.Row():
        with gr.Column():
            steps = gr.Slider(100, 500, step=50, value=300, label="Steps (more = better quality, slower)")
        with gr.Column():
            alpha = gr.Slider(0.1, 10, step=0.1, value=1, label="Content Weight")
        with gr.Column():
            beta = gr.Slider(1e4, 1e7, step=1e4, value=1e6, label="Style Weight")

    btn = gr.Button("Generate Stylized Image", variant="primary", size="lg")

    with gr.Row():
        output_img = gr.Image(type="pil", label="Stylized Output", height=400, elem_classes="card-section")

    btn.click(fn=run_nst, inputs=[content_img, style_img, steps, alpha, beta], outputs=output_img)

demo.launch()
