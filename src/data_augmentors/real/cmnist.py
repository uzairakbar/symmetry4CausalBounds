import torch
import numpy as np
from numpy.typing import NDArray
from torchvision import transforms
from abc import ABC, abstractmethod
from torchvision.transforms import functional as F
from typing import Dict, Literal, List, Optional, Tuple

from src.data_augmentors.abstract import DataAugmenter as DA


class StandardScaler():
    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, sample):
        return (sample - self.mean)/self.std


class UniformStandardScaler(StandardScaler):
    def __init__(self, low, high):
        self.mean = (low + high)/2.0
        self.std = (high - low)/(12.0**0.5)


ALPHA, BETA = 2, 2
class BetaStandardScaler(StandardScaler):
    def __init__(self, min, max, a=ALPHA, b=BETA):
        self.mean = a/(a + b)
        self.std = np.sqrt(a*b)/((a+b) * np.sqrt(a+b+1))
        self.min = min
        self.max = max
    
    def rescale(self, sample):
        return sample*(self.max-self.min) + self.min
    
    def __call__(self, sample):
        sample = (sample-self.min)/(self.max-self.min)
        return super(BetaStandardScaler, self).__call__(sample=sample)


class Compose(transforms.Compose):
    def __init__(self, transforms):
        super(Compose, self).__init__(transforms)

    def __call__(self, img):
        params = tuple()
        for t in self.transforms:
            img, transform_params = t(img)
            params += transform_params
        return img, params


class Jitter(transforms.ColorJitter):
    def __init__(
        self,
        brightness = 0,
        contrast = 0,
        saturation = 0,
        hue = 0
    ):
        self.brightness_scaler = BetaStandardScaler(
            max(0, 1 - brightness), 1 + brightness
        )
        self.contrast_scaler = BetaStandardScaler(
            max(0, 1 - contrast), 1 + contrast
        )
        self.saturation_scaler = BetaStandardScaler(
            max(0, 1 - saturation), 1 + saturation
        )
        self.hue_scaler = BetaStandardScaler(-1*hue, hue)
        return super(Jitter, self).__init__(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    @property
    def augmentation(self):
        return 'jitter'
    
    def get_params(
        self
    ):
        b = float( torch.distributions.beta.Beta(ALPHA, BETA).sample() )
        c = float( torch.distributions.beta.Beta(ALPHA, BETA).sample() )
        s = float( torch.distributions.beta.Beta(ALPHA, BETA).sample() )
        h = float( torch.distributions.beta.Beta(ALPHA, BETA).sample() )
        b = self.brightness_scaler.rescale(b)
        c = self.contrast_scaler.rescale(c)
        s = self.saturation_scaler.rescale(s)
        h = self.hue_scaler.rescale(h)
        return b, c, s, h

    def forward(self, img):
        b, c, s, h = self.get_params()
        img = F.adjust_brightness(img, b)
        img = F.adjust_contrast(img, c)
        img = F.adjust_saturation(img, s)
        img = F.adjust_hue(img, h)
        scaled_b = self.brightness_scaler(b)
        scaled_c = self.contrast_scaler(c)
        scaled_s = self.saturation_scaler(s)
        scaled_h = self.hue_scaler(h)
        return img, (
            scaled_b,scaled_c,scaled_s,scaled_h
        )


class Translate(transforms.RandomAffine):
    def __init__(self, translate = (0.0, 0.0)):
        dx, dy = translate
        self.param_scaler = (BetaStandardScaler(-1*dx, dx),
                             BetaStandardScaler(-1*dy, dy))
        return super(Translate, self).__init__(0,
                                               translate=translate,
                                               scale=None,
                                               shear=None,
                                               interpolation=transforms.InterpolationMode.BILINEAR,
                                               fill=0)
    
    @property
    def augmentation(self):
        return 'translate'
    
    def get_params(
        self,
        degrees,
        img_size,
    ):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        
        tx = float( torch.distributions.beta.Beta(ALPHA, BETA).sample() )
        ty = float( torch.distributions.beta.Beta(ALPHA, BETA).sample() )
        tx = self.param_scaler[0].rescale(tx)
        ty = self.param_scaler[1].rescale(ty)
        tx = int(round(tx * img_size[0]))
        ty = int(round(tx * img_size[1]))
        translations = (tx, ty)

        scale = 1.0
        shear_x = shear_y = 0.0
        shear = (shear_x, shear_y)

        return angle, translations, scale, shear
    
    def forward(self, img):    
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F.get_image_size(img)

        ret = self.get_params(self.degrees, img_size)
        params = ret[1]
        
        img = F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)
        scaled_params = tuple(param_scaler(param) for (param_scaler, param) in zip(self.param_scaler, params))
        return img, scaled_params


JITTER_PARAM = 0.5
TRANSLATE_PARAM = 0.2
Augmentation = Literal['jitter', 'translate']
ALL_AUGMENTATIONS: Dict[Augmentation, DA] = {
    augmenter.augmentation: augmenter for augmenter in ([
        Jitter(
            JITTER_PARAM, JITTER_PARAM, JITTER_PARAM, JITTER_PARAM
        ),
        Translate(
            (TRANSLATE_PARAM, TRANSLATE_PARAM)
        ),
    ])
}


class ColoredDigitsDA(DA):
    def __init__(
            self,
            augmentations: Optional[str]=None
        ):
        # self.to_pil = transforms.ToPILImage(mode='RGB')
        # self.to_tensor = transforms.ToTensor()

        if augmentations:
            augmentations: List[Augmentation] = augmentations.replace(' ','').split('>')
        else:
            augmentations: List[Augmentation] = list(ALL_AUGMENTATIONS.keys())
        
        self._augmentations: List[DA] = Compose([
            ALL_AUGMENTATIONS[augmentation] for augmentation in augmentations
        ])
    
    @property
    def augmentation(self):
        return 'colored_mnist'
    
    def augment(
            self,
            X: NDArray
        ) -> Tuple[NDArray, NDArray]:
        GX, G = [], []
        for image in X:
            gx, g = self._augmentations(torch.tensor(image))
            GX.append(gx.numpy())
            G.append(np.array(g))
        GX = np.stack(GX)
        G = np.stack(G)
        return GX, G