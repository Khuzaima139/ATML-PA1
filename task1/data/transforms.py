import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch

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

def make_permutations(n_images, n_patches, seed):
    generator = torch.Generator().manual_seed(seed)
    perms = []
    for _ in range(n_images):
        while True:
            p = torch.randperm(n_patches, generator=generator)
            if not torch.equal(p, torch.arange(n_patches)):
                break
        perms.append(p)
    return torch.stack(perms)


def patch_shuffle(x, perms, grid):
    b, c, h, w = x.shape
    ph, pw = h // grid, w // grid
    patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
    patches = patches.reshape(b, c, grid * grid, ph, pw)
    out = torch.empty_like(patches)
    for i in range(b):
        out[i] = patches[i, :, perms[i]]
    out = out.reshape(b, c, grid, grid, ph, pw)
    out = out.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
    return out