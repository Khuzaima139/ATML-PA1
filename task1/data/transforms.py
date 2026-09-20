import torchvision.transforms.functional as TF


def grayscale(x):
    return TF.rgb_to_grayscale(x, num_output_channels=3)


def hue_rotate(x, factor):
    return TF.adjust_hue(x, factor)