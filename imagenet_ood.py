import logging
import os
from typing import Callable, Optional

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg

_logger = logging.getLogger(__name__)


class ImageNetA(ImageFolder):
    """ImageNetA dataset.

    - Paper: [https://arxiv.org/abs/1907.07174](https://arxiv.org/abs/1907.07174).
    """

    base_folder = "imagenet-a"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
    filename = "imagenet-a.tar"
    tgz_md5 = "c3e55429088dc681f30d81f4726b6595"

    def __init__(self, root: str, split=None, transform: Optional[Callable] = None, download: bool = False, **kwargs):
        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        super().__init__(root=os.path.join(root, self.base_folder), transform=transform, **kwargs)

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def _check_integrity(self) -> bool:
        return check_integrity(os.path.join(self.root, self.filename), self.tgz_md5)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            _logger.debug("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


class ImageNetO(ImageNetA):
    """ImageNetO datasets.

    Contains unknown classes to ImageNet-1k.


    - Paper: [https://arxiv.org/abs/1907.07174](https://arxiv.org/abs/1907.07174)
    """

    base_folder = "imagenet-o"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"
    filename = "imagenet-o.tar"
    tgz_md5 = "86bd7a50c1c4074fb18fc5f219d6d50b"


class ImageNetR(ImageNetA):
    """ImageNet-R(endition) dataset.

    Contains art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings,
    patterns, plastic objects,plush objects, sculptures, sketches, tattoos, toys,
    and video game renditions of ImageNet-1k classes.

    - Paper: [https://arxiv.org/abs/2006.16241](https://arxiv.org/abs/2006.16241)
    """

    base_folder = "imagenet-r"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    filename = "imagenet-r.tar"
    tgz_md5 = "a61312130a589d0ca1a8fca1f2bd3337"


class NINCOFull(ImageFolder):
    """`NINCO` Dataset subset.

    Args:
        root (string): Root directory of dataset where directory
            exists or will be saved to if download is set to True.
        split (string, optional): The dataset split, not used.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, `transforms.RandomCrop`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        **kwargs: Additional arguments passed to :class:`~torchvision.datasets.ImageFolder`.
    """

    PAPER_URL = "https://arxiv.org/pdf/2306.00826.pdf"
    base_folder = "ninco"
    filename = "NINCO_all.tar.gz"
    file_md5 = "b9ffae324363cd900a81ce3c367cd834"
    url = "https://zenodo.org/record/8013288/files/NINCO_all.tar.gz"
    # size: 15393

    def __init__(
        self, root: str, split=None, transform: Optional[Callable] = None, download: bool = False, **kwargs
    ) -> None:
        self.root = os.path.expanduser(root)
        self.dataset_folder = os.path.join(self.root, self.base_folder)
        self.archive = os.path.join(self.root, self.filename)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        super().__init__(self.dataset_folder, transform=transform, **kwargs)

    def _check_integrity(self) -> bool:
        return check_integrity(self.archive, self.file_md5)

    def _check_exists(self) -> bool:
        return os.path.exists(self.dataset_folder)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        download_and_extract_archive(
            self.url, download_root=self.root, extract_root=self.dataset_folder, md5=self.file_md5
        )


if __name__ == "__main__":
    ImageNetR(root="data", download=True)
    ImageNetO(root="data", download=True)
    ImageNetA(root="data", download=True)
    NINCOFull(root="data", download=True)
