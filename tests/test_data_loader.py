import unittest

from utils.file_utils import get_base_id


class DataLoaderBaseIdTests(unittest.TestCase):
    def test_get_base_id_preserves_street_view_orientation_suffix(self):
        self.assertEqual(
            get_base_id("0Qps9neACpF8zhb7zAmyGg_180_Blank.png"),
            "0Qps9neACpF8zhb7zAmyGg_180",
        )

    def test_get_base_id_preserves_internal_underscores(self):
        self.assertEqual(
            get_base_id("tmpbDw7gUG1aNIz7M6hp0g_180_adversarial_3.png"),
            "tmpbDw7gUG1aNIz7M6hp0g_180",
        )


if __name__ == "__main__":
    unittest.main()
