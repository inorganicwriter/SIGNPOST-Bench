import unittest

import config


class ConfigProviderTests(unittest.TestCase):
    def test_config_exposes_only_sponsor_and_gemini_api_keys(self):
        self.assertTrue(hasattr(config, "SPONSOR_API_KEY"))
        self.assertTrue(hasattr(config, "GEMINI_API_KEY"))
        self.assertFalse(hasattr(config, "OPENROUTER_API_KEY"))
        self.assertFalse(hasattr(config, "SILICONFLOW_API_KEY"))
        self.assertFalse(hasattr(config, "OPENAI_API_KEY"))
        self.assertFalse(hasattr(config, "RELAY_API_KEY"))

    def test_get_api_key_rejects_removed_providers(self):
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            config.get_api_key("openai")


if __name__ == "__main__":
    unittest.main()
