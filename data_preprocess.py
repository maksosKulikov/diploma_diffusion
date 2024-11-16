from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_data(photo_size, batch_size, main_link):
    TRANSFORM = transforms.Compose([transforms.Resize(photo_size),
                                    transforms.ToTensor()])

    train_data = ImageFolder(root=main_link + "\\train", transform=TRANSFORM)
    test_data = ImageFolder(root=main_link + "\\test", transform=TRANSFORM)
    val_data = ImageFolder(root=main_link + "\\val", transform=TRANSFORM)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    print(f"Количество батчей в train = {len(train_loader)}")
    return train_loader, val_loader, test_loader
