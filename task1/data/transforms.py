import torchvision.transforms.functional as TF
import torch.nn.functional as F

def grayscale(x):
    return TF.rgb_to_grayscale(x, num_output_channels=3)


def hue_rotate(x, factor):
    return TF.adjust_hue(x, factor)

def translate(x, delta, direction):
    if delta == 0:
        return x
    padded = F.pad(x, (delta, delta, delta, delta), mode="reflect")
    offsets = {
        "right": (delta, 0),
        "left": (delta, 2 * delta),
        "down": (0, delta),
        "up": (2 * delta, delta),
    }
    row_off, col_off = offsets[direction]
    h, w = x.shape[-2:]
    return padded[..., row_off:row_off + h, col_off:col_off + w]