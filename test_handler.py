import base64
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

import handler


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Right90PrepTests(unittest.TestCase):
    def setUp(self):
        self._upload_jpeg = handler.upload_jpeg
        self._upload_webp_lossless = handler.upload_webp_lossless
        handler.upload_jpeg = lambda _img, filename, quality=95: f"https://example.test/{filename}"
        handler.upload_webp_lossless = lambda _img, filename: f"https://example.test/{filename}"
    def test_right90_visia_ready_without_cheek_fragment(self):
        img = Image.new("RGB", (2160, 2700), (40, 80, 120))

        result = handler.process_images({"right_90": image_to_b64(img)}, {}, "redness")["right_90"]

        self.assertEqual(result["analysis_step"], "visia_ready")
        self.assertIn("visia_image_url", result)
        self.assertNotIn("cheek_roi_image_url", result)

    def test_handler_only_requires_right90_image(self):
        img = Image.new("RGB", (1080, 1920), (40, 80, 120))

        result = handler.handler({"input": {"images": {"right_90": image_to_b64(img)}}})

        self.assertEqual(result["status"], "visia_ready")
        self.assertIn("right_90", result["processed_angles"])


if __name__ == "__main__":
    unittest.main()
