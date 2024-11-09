#%%
import numpy as np
import torch, cv2



def white_balance(
    image, black_level=2045, white_level=15092, red_gain=2.156868, green_gain=0.940327, blue_gain=1.286982,
    subtract_black_level = True,
) -> np.ndarray:
    
    if subtract_black_level:
        image = (image - black_level) / (white_level - black_level)
    else:
        image = image / (white_level - black_level)
    # image = np.clip(image, 0, 1)
    coeffs = np.array([red_gain, green_gain, blue_gain])

    image = image * coeffs

    image = np.clip(image, 0, 1)

    return image

def tone_map_torch(image: torch.Tensor, K: float = 0.15, B: float = 0.95) -> torch.Tensor:

    if image.dim() == 3:    
        height, width, _ = image.shape
        N = height * width
        I_m = torch.exp(torch.sum(torch.log(image + 1e-5), axis=(0, 1)) / N)
    elif image.dim() == 2:
        N = image.size()[0]
        I_m = torch.exp(torch.sum(torch.log(image + 1e-5), axis=(0)) / N)
    else:
        raise ValueError(f'Unknown image dim {image.dim}. Must be 2 or 3.')

    # print(K/I_m, 'K/I_m')
    tilde_I = K / I_m * image
    I_white = B * torch.max(tilde_I)
    # print(I_white, 'I_white')

    return tilde_I * (1 + tilde_I / I_white**2) / (1 + tilde_I)


def tone_map_hardcoded_torch(image: torch.Tensor, mode: str = 'flash') -> torch.Tensor:
    
    # info(image, 'image')

    if 'flash' in mode.lower():
        tilde_I = image * torch.tensor([3.15342155, 4.75500978, 6.73902118]).to(image.device)
        I_white = 2.4737937681395112
    elif 'noflash' in mode.lower():
        tilde_I = image * torch.tensor([4.87251765,  7.60801591, 12.21575474]).to(image.device)
        I_white = 2.437383494442701
    elif 'diff' in mode.lower():
        tilde_I = image * torch.tensor([8.16718901, 11.23840394, 14.13109103]).to(image.device)
        I_white = 5.002156378483877
    else:
        raise ValueError(f'Unknown mode {mode}')

    return tilde_I * (1 + tilde_I / I_white**2) / (1 + tilde_I)


def gamma_correct_torch(image: torch.Tensor) -> torch.Tensor:
    mask = image <= 0.0031308
    image[mask] *= 12.92
    image[~mask] = (1 + 0.055) * image[~mask] ** (1 / 2.5) - 0.055
    return image


def isp(image: np.ndarray, subtract_black_level=True, tonemap_mode='', black_level=2045, white_level=15092, red_gain=2.156868, green_gain=0.940327, blue_gain=1.286982):

    image = white_balance(image, black_level, white_level, red_gain, green_gain, blue_gain, subtract_black_level=subtract_black_level)

    # plotinfo(image*15, f'{tonemap_mode} white_balance')
    if tonemap_mode == '' or tonemap_mode == 'none' or tonemap_mode is None or tonemap_mode == 'general':
        image = tone_map_torch(image)
    else:
        image = tone_map_hardcoded_torch(image, mode=tonemap_mode)
    
    image = gamma_correct_torch(image)
    return image





#%%

def load_RAW_image(image, img_wh=(1200, 800), subtract_black_level=True):

    # output is a torch tensor, H*W, 3
    # output is [0-1] * 15
    # img = np.load(image_path).astype(np.float32)
    img = image.astype(np.float32)
    img = white_balance(img, subtract_black_level=subtract_black_level)
    img = img * 1

    img = cv2.resize(img, img_wh, interpolation=cv2.INTER_AREA)

    # img = torch.from_numpy(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB

    # img = img ** .22
    # img = img * 15
    return img


def render_RAW_image(img, mode='flash'):

    # input should be a torch tensor, the last dimension must be 3
    # input scale is [0-1] * 15
    # output is a torch tensor, H*W, 3, scale is [0-1]

    # img = img ** (1/.22)

    img = img.permute(1, 2, 0)

    img = img / 1

    if mode == 'flash':
        img = tone_map_hardcoded_torch(img, mode='flash')
    elif mode == 'noflash':
        img = tone_map_hardcoded_torch(img, mode='noflash')
    elif mode == 'diff':
        img = tone_map_hardcoded_torch(img, mode='diff')
    else:
        img = tone_map_torch(img)
    
    img = gamma_correct_torch(img)

    img = img.permute(2, 0, 1)
    return img
