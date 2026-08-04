import unittest

from utils.parsers import parse_defense_response, parse_json_response


class ParserTests(unittest.TestCase):
    def test_parse_json_response_handles_code_blocks_and_surrounding_text(self):
        text = 'Reasoning\n```json\n{"consistent": false, "trusted_source": "Visual"}\n```'

        self.assertEqual(
            parse_json_response(text),
            {"consistent": False, "trusted_source": "Visual"},
        )

    def test_parse_json_response_ignores_thinking_tags(self):
        text = '<think>hidden</think>{"final_prediction": "(1.0, 2.0)"}'

        self.assertEqual(
            parse_json_response(text),
            {"final_prediction": "(1.0, 2.0)"},
        )

    def test_parse_defense_response_falls_back_to_coordinates(self):
        parsed = parse_defense_response("I trust visual evidence. Final answer: (35.68, 139.76)")

        self.assertEqual(parsed["final"], "(35.68, 139.76)")


if __name__ == "__main__":
    unittest.main()
