"""Tests src.data.transforms"""
import os
import unittest

import cv2
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from src.data.transforms import BlurGray, ClassicSegMean, ClassicSegOtsu


class ClassicSegTestCase(unittest.TestCase):
    """Class to run tests on BlurGray, ClassicSegMean, ClassicSegOtsu in src.data.transforms"""

    @classmethod
    def setUpClass(cls):
        cls.resource_dir = "tests/helpers/resources/"
        # Load Dataloader with ClassicSegMean transform
        config_name = os.path.join(
            cls.resource_dir,
            "configs/datamodule/default-object-detection-aug-classic-seg-mean.yaml",
        )
        args = OmegaConf.load(config_name)
        cls.datamodule_mean: LightningDataModule = instantiate(
            args, data_config=args, _recursive_=False
        )

        # Load Dataloader with ClassicSegOtsu transform
        config_name = os.path.join(
            cls.resource_dir,
            "configs/datamodule/default-object-detection-aug-classic-seg-otsu.yaml",
        )
        args = OmegaConf.load(config_name)
        cls.datamodule_otsu: LightningDataModule = instantiate(
            args, data_config=args, _recursive_=False
        )

        # Get Base Model Config (this doesn't include metric callback, not required to train model)
        config_name = os.path.join(
            cls.resource_dir, "configs/model/ssd-nvidia/test-no-lr-3-channel.yaml"
        )
        args = OmegaConf.load(config_name)
        cls.model: LightningModule = instantiate(
            args.model, model_config=args["model"], _recursive_=False
        )

    def _test_augment_return(self, out):
        """Test for return length and types"""

        self.assertEqual(len(out), 5, "The ouput should be a 5 tuple")
        with self.subTest():
            self.assertIsInstance(
                out[0], np.ndarray, "The first output should be a numpy array of the image"
            )
        with self.subTest():
            self.assertEqual(
                out[1:],
                (None, None, None, None),
                "All output variables other than image should be None",
            )

    def _test_image_pixels(self, img):
        """Test for range of pixel values"""
        with self.subTest():
            self.assertLessEqual(np.amax(img), 255.0, "Image has pixel value higher than 255")
        with self.subTest():
            self.assertGreaterEqual(np.amin(img), 0.0, "Image has pixel value lower than 0")

    def _test_blur_gray_base(self):
        """Blurgray: Test to call for return, range and tranform shape tests"""
        test_img_dims = (512, 512, 3)
        test_img_low = 0
        test_img_high = 255
        test_img_blur = 5

        test_img = np.random.randint(
            low=test_img_low, high=test_img_high, size=test_img_dims
        ).astype(np.uint8)
        blur_gray = BlurGray(test_img_blur)
        gray_out = blur_gray(test_img)
        self._test_augment_return(gray_out)
        with self.subTest():
            self._test_image_pixels(gray_out[0])
        with self.subTest():
            self.assertEqual(
                gray_out[0].shape,
                test_img_dims[:2],
                f"The output shape is wrong. \n Expected {test_img_dims[:2]}; got"
                f" {gray_out[0].shape}",
            )

    def _test_blur_gray_black(self):
        """BlurGray: Test grayscale conversion individually"""
        test_img_dims = (2, 2, 3)
        test_img_blur = 0
        test_img = np.random.randint(low=0, high=255, size=test_img_dims).astype(np.uint8)
        exp_gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        blur_gray = BlurGray(test_img_blur)
        gray_img = blur_gray(test_img)[0]
        self.assertTrue(
            (gray_img == exp_gray_img).all(),
            "Black input image should return the same image when 0 blur",
        )

    def _test_blur_gray_match(self):
        """BlurGray: Test output with ground truth"""
        test_img_blur = 10
        test_img_path = os.path.join(self.resource_dir, "images/pest.jpg")
        test_img = np.asarray(Image.open(test_img_path))
        blur_gray = BlurGray(test_img_blur)
        gray_img = blur_gray(test_img)[0]
        exp_gray_img_path = os.path.join(
            self.resource_dir,
            "pkls",
            f"{os.path.splitext(os.path.basename(test_img_path))[0]}_gray_{test_img_blur}.npy",
        )
        exp_gray_img = np.load(exp_gray_img_path)
        self.assertTrue(
            (gray_img == exp_gray_img).all(),
            "The blur gray output should match the expected ground truth",
        )

    def _test_classic_seg_base(self, ClassicSeg):
        """ClassicSeg: Test to call for return, range, tranform shape and invariance tests"""
        test_img_dims = (2, 2, 3)
        test_img_low = 0
        test_img_high = 255
        test_img_blur = 5

        test_img = np.random.randint(
            low=test_img_low, high=test_img_high, size=test_img_dims
        ).astype(np.uint8)
        classic_seg = ClassicSeg(test_img_blur)
        seg_out = classic_seg(test_img)

        self._test_augment_return(seg_out)
        with self.subTest():
            self._test_image_pixels(seg_out[0])

        exp_seg_img_dims = (*test_img_dims[:-1], test_img_dims[-1] + 1)

        with self.subTest():
            self.assertEqual(
                seg_out[0].shape,
                exp_seg_img_dims,
                f"The output shape is wrong. \n Expected {exp_seg_img_dims}; got"
                f" {seg_out[0].shape}",
            )
        with self.subTest():
            self.assertTrue(
                (test_img == seg_out[0][..., : test_img_dims[-1]]).all(),
                "The original channels of the image have been altered.",
            )

    def _test_classic_seg_black(self, ClassicSeg):
        """ClassicSeg: Test black image on no blur"""
        test_img_dims = (2, 2, 3)
        test_img_blur = 0
        test_img = np.zeros(test_img_dims).astype(np.uint8)
        classic_seg = ClassicSeg(test_img_blur)
        seg_channel = classic_seg(test_img)[0][..., -1]
        self.assertTrue(
            (seg_channel == (np.ones(seg_channel.shape) * 255).astype(np.uint8)).all(),
            "Black input image should return white image when 0 blur",
        )

    def _test_classic_seg_match(self, ClassicSeg, seg_name):
        """ClassicSeg: Test output with ground truth"""
        test_img_blur = 10
        test_img_path = os.path.join(self.resource_dir, "images/pest.jpg")
        test_img = np.asarray(Image.open(test_img_path))
        classic_seg = ClassicSeg(test_img_blur)
        seg_channel = classic_seg(test_img)[0][..., -1]
        exp_seg_channel_basename = (
            f"{os.path.splitext(os.path.basename(test_img_path))[0]}_{seg_name}_{test_img_blur}.npy"
        )
        exp_seg_channel_path = os.path.join(
            self.resource_dir,
            "pkls",
            exp_seg_channel_basename,
        )
        exp_seg_channel = np.load(exp_seg_channel_path)
        self.assertTrue(
            (seg_channel == exp_seg_channel).all(),
            "The segmentation output should match the expected ground truth",
        )

    def _test_integration_classic_seg(self, datamodule):
        """ClassicSeg: Test integration on fast dev run"""
        trainer = Trainer(
            fast_dev_run=True,
            logger=False,
            checkpoint_callback=False,
            progress_bar_refresh_rate=0,
            weights_summary=None,
        )
        try:
            trainer.fit(self.model, datamodule)
        except Exception as e:
            self.fail(f"Fast dev run failed with the following exception:\n{e}")

    def test_blur_gray(self):
        """BlurGray: Test to call all tests"""
        with self.subTest():
            self._test_blur_gray_base()
        with self.subTest():
            self._test_blur_gray_black()
        with self.subTest():
            self._test_blur_gray_match()

    def test_classic_seg_mean(self):
        """ClassicSegMean: Test to call all tests"""
        ClassicSeg = ClassicSegMean
        seg_name = "mean"
        with self.subTest():
            self._test_classic_seg_base(ClassicSeg)
        with self.subTest():
            self._test_classic_seg_black(ClassicSeg)
        with self.subTest():
            self._test_classic_seg_match(ClassicSeg, seg_name)

    def test_classic_seg_otsu(self):
        """ClassicSegOtsu: Test to call all tests"""
        ClassicSeg = ClassicSegOtsu
        seg_name = "otsu"
        with self.subTest():
            self._test_classic_seg_base(ClassicSeg)
        with self.subTest():
            self._test_classic_seg_black(ClassicSeg)
        with self.subTest():
            self._test_classic_seg_match(ClassicSeg, seg_name)

    def test_integration_classic_seg_mean(self):
        """ClassicSegMean: Test to call integration test"""
        datamodule = self.datamodule_mean
        self._test_integration_classic_seg(datamodule)

    def test_integration_classic_seg_otsu(self):
        """ClassicSegOtsu: Test to call integration test"""
        datamodule = self.datamodule_otsu
        self._test_integration_classic_seg(datamodule)


if __name__ == "__main__":

    unittest.main()
