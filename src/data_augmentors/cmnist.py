from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torchvision import transforms
from torchvision.transforms import functional as F

from src.data_augmentors.abstract import DataAugmenter as DA
from src.data_augmentors.utils import BetaStandardScaler

ALPHA, BETA = 2, 2
JITTER_PARAM = 0.5
TRANSLATE_PARAM = 0.2


class Compose(transforms.Compose):
    """Compose transforms that return both output and parameters."""

    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img):
        params = tuple()
        for t in self.transforms:
            img, transform_params = t(img)
            params += transform_params
        return img, params


class Jitter(transforms.ColorJitter):
    """Color jitter with standardized parameters."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness_scaler = BetaStandardScaler(max(0, 1 - brightness), 1 + brightness)
        self.contrast_scaler = BetaStandardScaler(max(0, 1 - contrast), 1 + contrast)
        self.saturation_scaler = BetaStandardScaler(max(0, 1 - saturation), 1 + saturation)
        self.hue_scaler = BetaStandardScaler(-1 * hue, hue)
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    @property
    def augmentation(self):
        return "jitter"

    def get_params(self):
        """Sample parameters from Beta distribution."""
        b = float(torch.distributions.beta.Beta(ALPHA, BETA).sample())
        c = float(torch.distributions.beta.Beta(ALPHA, BETA).sample())
        s = float(torch.distributions.beta.Beta(ALPHA, BETA).sample())
        h = float(torch.distributions.beta.Beta(ALPHA, BETA).sample())

        b = self.brightness_scaler.rescale(b)
        c = self.contrast_scaler.rescale(c)
        s = self.saturation_scaler.rescale(s)
        h = self.hue_scaler.rescale(h)

        return b, c, s, h

    def forward(self, img):
        """Apply jitter and return standardized parameters."""
        b, c, s, h = self.get_params()

        img = F.adjust_brightness(img, b)
        img = F.adjust_contrast(img, c)
        img = F.adjust_saturation(img, s)
        img = F.adjust_hue(img, h)

        scaled_b = self.brightness_scaler(b)
        scaled_c = self.contrast_scaler(c)
        scaled_s = self.saturation_scaler(s)
        scaled_h = self.hue_scaler(h)

        return img, (scaled_b, scaled_c, scaled_s, scaled_h)


class Translate(transforms.RandomAffine):
    """Translation with standardized parameters."""

    def __init__(self, translate=(0.0, 0.0)):
        dx, dy = translate
        self.param_scaler = (BetaStandardScaler(-1 * dx, dx), BetaStandardScaler(-1 * dy, dy))
        super().__init__(
            0, translate=translate, scale=None, shear=None, interpolation=transforms.InterpolationMode.BILINEAR, fill=0
        )

    @property
    def augmentation(self):
        return "translate"

    def get_params(self, degrees, img_size):
        """Sample translation parameters."""
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())

        tx = float(torch.distributions.beta.Beta(ALPHA, BETA).sample())
        ty = float(torch.distributions.beta.Beta(ALPHA, BETA).sample())

        tx = self.param_scaler[0].rescale(tx)
        ty = self.param_scaler[1].rescale(ty)

        tx = int(round(tx * img_size[0]))
        ty = int(round(ty * img_size[1]))
        translations = (tx, ty)

        scale = 1.0
        shear_x = shear_y = 0.0
        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def forward(self, img):
        """Apply translation and return standardized parameters."""
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


Augmentation = Literal["jitter", "translate"]

ALL_AUGMENTATIONS: dict[Augmentation, DA] = {
    augmenter.augmentation: augmenter
    for augmenter in [
        Jitter(JITTER_PARAM, JITTER_PARAM, JITTER_PARAM, JITTER_PARAM),
        Translate((TRANSLATE_PARAM, TRANSLATE_PARAM)),
    ]
}


class ColoredDigitsDA(DA):
    """Data augmenter for colored MNIST digits."""

    def __init__(self, augmentations: str | None = None):
        if augmentations:
            augmentations: list[Augmentation] = augmentations.replace(" ", "").split(">")
        else:
            augmentations: list[Augmentation] = list(ALL_AUGMENTATIONS.keys())

        self._augmentations: list[DA] = Compose([ALL_AUGMENTATIONS[augmentation] for augmentation in augmentations])

    @property
    def augmentation(self):
        return "colored_mnist"

    def augment(self, X: NDArray) -> tuple[NDArray, NDArray]:
        """Apply augmentations to batch of images."""
        GX, G = [], []
        for image in X:
            gx, g = self._augmentations(torch.tensor(image))
            GX.append(gx.numpy())
            G.append(np.array(g))

        GX = np.stack(GX)
        G = np.stack(G)

        return GX, G
