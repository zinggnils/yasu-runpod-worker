import base64
import unittest
from io import BytesIO

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
        self._upload_png = handler.upload_png
        handler.upload_jpeg = lambda _img, filename, quality=95: f"https://example.test/{filename}"
        handler.upload_webp_lossless = lambda _img, filename: f"https://example.test/{filename}"
        handler.upload_png = lambda _img, filename: f"https://example.test/{filename}"

    def tearDown(self):
        handler.upload_jpeg = self._upload_jpeg
        handler.upload_webp_lossless = self._upload_webp_lossless
        handler.upload_png = self._upload_png

    def test_right90_cheek_roi_outputs(self):
        img = Image.new("RGB", (2160, 2700), (185, 126, 104))

        result = handler.process_images({"right_90": image_to_b64(img)}, {}, "redness")["right_90"]

        self.assertEqual(result["analysis_step"], "cheek_roi")
        self.assertIn(
            result["cheek_roi_method"],
            ("clipseg_angular_fragment_bone", "clipseg_fallback_alpha"),
        )
        self.assertIn("cheek_roi_image_url", result)
        self.assertTrue(result["cheek_roi_image_url"].endswith(".png"))
        self.assertIn("visia_image_url", result)
        self.assertIn("clean_image_url", result)
        self.assertNotIn("redness_score", result)
        self.assertGreaterEqual(result["cheek_pixel_count"], 0)

    def test_handler_only_requires_right90_image(self):
        img = Image.new("RGB", (1080, 1920), (210, 150, 140))

        result = handler.handler({"input": {"images": {"right_90": image_to_b64(img)}}})

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["analysis_step"], "cheek_roi")
        self.assertIn("right_90", result["analysis_angles"])
        self.assertEqual(
            result["processed_angles"]["right_90"]["analysis_step"], "cheek_roi"
        )


if __name__ == "__main__":
    unittest.main()
