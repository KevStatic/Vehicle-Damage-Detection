import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataset import VehicleDamageDataset
from model import get_model
import torchvision.transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def train():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    print("Loading datasets...")
    dataset = VehicleDamageDataset(
        'Datasets/coco/train', 'Datasets/coco/train/COCO_train_annos.json', transforms=get_transform(True))
    dataset_val = VehicleDamageDataset(
        'Datasets/coco/val', 'Datasets/coco/val/COCO_val_annos.json', transforms=get_transform(False))
    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(dataset_val)}")

    data_loader = DataLoader(dataset, batch_size=2,
                             shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = 2  # background + damage
    print(f"Building model with {num_classes} classes...")
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            batch_count += 1

        lr_scheduler.step()
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Saving model...")
    torch.save(model.state_dict(), 'vehicle_damage_model.pth')
    print("Training complete! Model saved as 'vehicle_damage_model.pth'")


if __name__ == "__main__":
    train()
