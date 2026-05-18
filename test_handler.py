import base64
import unittest
from io import BytesIO

from PIL import Image

import handler


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="WEBP", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Right90AnalysisTests(unittest.TestCase):
    def test_normalizes_and_crops_to_contract_sizes(self):
        img = Image.new("RGB", (2160, 2700), (185, 126, 104))

        result = handler.process_right90(image_to_b64(img))

        self.assertEqual(result["portrait"].size, (2160, 2700))
        self.assertEqual(result["crop"].size, (1000, 1000))
        self.assertEqual(result["crop_box"], {"x": 580, "y": 850, "width": 1000, "height": 1000})
        self.assertGreaterEqual(result["quality_score"], 0)
        self.assertLessEqual(result["quality_score"], 100)

    def test_handler_only_requires_right90_image(self):
        img = Image.new("RGB", (1080, 1920), (210, 150, 140))

        result = handler.handler({"input": {"images": {"right_90": image_to_b64(img)}}})

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["analysis_angle"], "right_90")
        self.assertIn("right_90", result["processed_angles"])
        self.assertIn("redness_score", result["processed_angles"]["right_90"])
        self.assertIn("white_score", result["processed_angles"]["right_90"])


if __name__ == "__main__":
    unittest.main()
