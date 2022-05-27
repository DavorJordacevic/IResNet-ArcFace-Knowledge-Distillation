from torchvision import transforms
from PIL import Image
import collections
from torchvision.transforms import functional as F
import random
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class ResizedPad:
    def __init__(self, size, interpolation=Image.BILINEAR):
        """Resize and pad the image to the target size while keeping the w/h ratio.
        """
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        if isinstance(self.size, int):
            self.size = (self.size, self.size)
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        img_w, img_h = img.size
        h, w = self.size
        if img_w / img_h > w / h:
            resized_w = w
            resized_h = int(img_h / img_w * resized_w)
            padding = int((h - resized_h)/2)
            padding = (0, padding, 0, h - resized_h - padding)
        else:
            resized_h = h
            resized_w = int(img_w / img_h * resized_h)
            padding = int((w - resized_w)/2)
            padding = (padding, 0, w - resized_w - padding, 0)
        resized_image = F.resize(img, (resized_h, resized_w), self.interpolation)
        img = F.pad(resized_image, padding)
        return img

    def __repr__(self):
        interpolate_str = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Use percentage rather than int.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, float):
            self.size = (size, size)
        else:
            self.size = size
        assert isinstance(self.size[0], float)
        assert isinstance(self.size[0], float)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        w, h = img.size
        output_size = (int(h * self.size[0]), int(w*self.size[1]))
        i, j, h, w = self.get_params(img, output_size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[1, 1, 1])
train_transform = lambda size: transforms.Compose([
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # transforms.RandomGrayscale(0.1),
    transforms.RandomRotation(10),
    RandomCrop((0.8, 0.8)),
    ResizedPad(size),
#    transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


test_transform = lambda size: transforms.Compose([
    ResizedPad(size),
    #transforms.Resize(size),
    transforms.ToTensor(),
    normalize
])