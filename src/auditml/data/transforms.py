from torchvision import transforms


def mnist_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def cifar_transform(train: bool = False):
    ops = []
    if train:
        ops.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return transforms.Compose(ops)
