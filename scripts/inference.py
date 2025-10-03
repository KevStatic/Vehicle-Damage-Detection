import torch
from PIL import Image
from torchvision.transforms import ToTensor
from model import get_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # background + damage
model = get_model(num_classes)
model.load_state_dict(torch.load('vehicle_damage_model.pth'))
model.to(device)
model.eval()

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    print(predictions)

if __name__ == "__main__":
    predict("Datasets/coco/val/1.jpg")
