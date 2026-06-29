from PIL import Image
import torch
import torchvision.transforms as transforms

def preprocess_image(image: Image.Image, target_size: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Transforms a raw PIL Image into a normalized PyTorch tensor ready for model input.
    Ensures single channel/grayscale images are mapped to 3-channel RGB.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform_pipeline = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Returns normalized tensor of shape [3, target_size[0], target_size[1]]
    return transform_pipeline(image)
