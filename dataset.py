import os

import torch
import torch.utils.data as data


def get_loader(root, is_train, batch_size=32, transform=None, is_cached=False):
    from torchvision.datasets import ImageFolder

    if is_train:
        if is_cached:
            dataset = NumpyDataset()
        else:
            dataset = ImageFolder(os.path.join(root, "train"), transform=transform)
    else:
        dataset = ImageDir(os.path.join(root, "test"))
        dataset = TransformDataset(dataset, transform=transform)

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=6)
    return loader


def process_inputs(inputs, model, n_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs = []
    for _ in range(n_epochs):
        for x, y in inputs:
            x = x.to(device)
            x = model(x)
            x = x.cpu().numpy()
            y = y.cpu().numpy()
            outputs.append((x, y))
            print(len(outputs))
    return outputs


def prepare_data(inputs, model, n_epochs=3):
    import numpy as np

    with torch.no_grad():
        d = process_inputs(inputs, model, n_epochs)

    x = [d for d, _ in d]
    y = [d.reshape(-1, 1) for _, d in d]
    np.save("x.npy", np.vstack(x))
    np.save("y.npy", np.vstack(y).reshape(-1))


class NumpyDataset(data.Dataset):

    def __init__(self):
        import numpy as np

        self.x = np.load("x.npy")
        self.y = np.load("y.npy")

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return self.x.shape[0]


class ItemClass:

    def __init__(self, name, items):
        self.name = name
        self.items = items
        self.subclasses = []

    def add_sub_class(self, sub_class):
        self.subclasses.append(sub_class)

    def flatten(self, level=0):
        return ItemClass.__flatten__(self, level, "")

    @staticmethod
    def __flatten__(root_class, level, prefix):
        # result can be a parameter to reduce array creation
        result = []
        if level == 0:
            # prefix = "{}/{}".format(prefix, root_class.name)
            prefix = root_class.name
        if level <= 0:
            for item in root_class.items:
                result.append((item, prefix))
        for sub_class in root_class.subclasses:
            result += ItemClass.__flatten__(sub_class, level - 1, prefix)
        return result


class ImageDir(data.Dataset):

    def __init__(self, root, loader=None, level=0):
        from torchvision.datasets.folder import IMG_EXTENSIONS
        from torchvision.datasets.folder import default_loader

        if loader:
            self.loader = loader
        else:
            self.loader = default_loader

        self.samples = ImageDir.make_samples(root, IMG_EXTENSIONS, level=level)

    def __getitem__(self, index):
        path, y = self.samples[index]
        x = self.loader(path)
        return x, os.path.basename(path)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def item_class(dir, extensions):
        from torchvision.datasets.folder import has_file_allowed_extension

        item_class = ItemClass(os.path.basename(dir), [])
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if os.path.isdir(d):
                item_class.add_sub_class(ImageDir.item_class(d, extensions))
            else:
                if has_file_allowed_extension(d, extensions):
                    item_class.items.append(d)
        return item_class

    @staticmethod
    def make_samples(dir, extensions, level):
        item_class = ImageDir.item_class(dir, extensions)
        return item_class.flatten(level=level)


class TransformDataset(data.Dataset):

    def __init__(self, dataset, transform=None, transform_target=None):
        self.dataset = dataset
        self.transform = transform
        self.transform_target = transform_target

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        if self.transform_target:
            y = self.transform_target
        return x, y

    def __len__(self):
        return len(self.dataset)
