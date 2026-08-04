import unittest

from evaluation import api_client


class ApiClientRegistryTests(unittest.TestCase):
    def test_provider_registry_excludes_ppapi(self):
        self.assertEqual(set(api_client.PROVIDER_CONFIGS), {"sponsor", "gemini"})
        self.assertNotIn("ppapi", api_client.DEFAULT_PROVIDER_TIMEOUTS)
        self.assertNotIn("ppapi", api_client.DEFAULT_PROVIDER_MAX_RETRIES)

    def test_model_registry_uses_only_supported_providers(self):
        unsupported = {
            name: entry.get("provider")
            for name, entry in api_client.MODEL_REGISTRY.items()
            if entry.get("provider") not in api_client.PROVIDER_CONFIGS
        }
        self.assertEqual(unsupported, {})

    def test_build_client_rejects_removed_provider_override(self):
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            api_client.build_client("gpt-4o", provider="ppapi", api_key="test-key")

    def test_parse_coordinates_handles_supported_formats(self):
        cases = [
            ('{"latitude": 35.6812, "longitude": 139.7671}', (35.6812, 139.7671)),
            ("Final: (48.8584, 2.2945)", (48.8584, 2.2945)),
            ("Latitude: -33.8688\nLongitude: 151.2093", (-33.8688, 151.2093)),
            ("Best guess 40.7128 -74.0060", (40.7128, -74.0060)),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(api_client.GeoLocalizationClient.parse_coordinates(text), expected)

    def test_parse_coordinates_rejects_out_of_range_values(self):
        self.assertEqual(
            api_client.GeoLocalizationClient.parse_coordinates("(120.0000, 200.0000)"),
            (None, None),
        )


if __name__ == "__main__":
    unittest.main()
