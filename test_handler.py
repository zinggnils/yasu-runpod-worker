import base64
import unittest
from io import BytesIO

from PIL import Image

import handler


def image_to_b64(img: Image.Image) -> str:
    # Tests inject inputs as PNG so we exercise the same lossless transport
    # the device now uses; Pillow auto-detects format from bytes on the other side.
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Right90AnalysisTests(unittest.TestCase):
    def setUp(self):
        self._upload_jpeg = handler.upload_jpeg
        self._upload_webp_lossless = handler.upload_webp_lossless
        self._upload_png = handler.upload_png
        handler.upload_jpeg = lambda _img, filename, quality=95: f"https://example.test/{filename}"
        handler.upload_webp_lossless = lambda _img, filename: f"https://example.test/{filename}"
        handler.upload_png = lambda _img, filename: f"https://example.test/{filename}"

    def tearDown(self):
        handler.upload_jpeg = self._upload_jpeg
        handler.upload_webp_lossless = self._upload_webp_lossless
        handler.upload_png = self._upload_png

    def test_right90_cheek_pct_scoring(self):
        img = Image.new("RGB", (2160, 2700), (185, 126, 104))

        result = handler.process_images({"right_90": image_to_b64(img)}, {}, "redness")["right_90"]

        self.assertIn("pct_ei", result["scoring_method"])
        self.assertIn("redness_image_url", result)
        self.assertTrue(result["redness_image_url"].endswith(".png"))
        self.assertNotIn("crop_image_url", result)
        self.assertGreaterEqual(result["redness_score"], 0)
        self.assertLessEqual(result["redness_score"], 100)

    def test_handler_only_requires_right90_image(self):
        img = Image.new("RGB", (1080, 1920), (210, 150, 140))

        result = handler.handler({"input": {"images": {"right_90": image_to_b64(img)}}})

        self.assertEqual(result["status"], "done")
        self.assertIn("right_90", result["analysis_angles"])
        self.assertIn("right_90", result["processed_angles"])
        self.assertIn("redness_score", result["processed_angles"]["right_90"])
        self.assertIn("white_score", result["processed_angles"]["right_90"])


if __name__ == "__main__":
    unittest.main()
