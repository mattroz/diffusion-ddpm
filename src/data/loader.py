from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader


def get_loader(config):
    dataset = load_dataset(config.dataset, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    return loader
