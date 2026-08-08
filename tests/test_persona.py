import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from roleplay.persona import (
    PersonaValidationError,
    load_persona,
    render_persona_prompt,
    validate_persona,
)


def persona(**overrides):
    data = {
        "name": "林遥",
        "identity": ["侦探"],
        "personality": ["冷静"],
        "speech_style": ["简短"],
        "relationships": [],
        "facts": [],
        "boundaries": ["不编造"],
    }
    data.update(overrides)
    return data


class PersonaValidationTests(unittest.TestCase):
    def test_arrays_can_be_empty_and_text_is_preserved(self):
        data = persona(identity=[" 原始措辞 "])
        self.assertIs(validate_persona(data), data)
        self.assertIn("-  原始措辞 ", render_persona_prompt(data))

    def test_rendered_prompt_prioritizes_readability_and_allows_creativity(self):
        prompt = render_persona_prompt(persona())
        self.assertIn("完整、自然、可读", prompt)
        self.assertIn("动作或神态描写是可选的", prompt)
        self.assertIn("合理创作", prompt)
        self.assertNotIn("必须遵循统一格式", prompt)

    def test_rejects_unknown_field(self):
        with self.assertRaisesRegex(PersonaValidationError, "未知字段"):
            validate_persona(persona(extra="no"))

    def test_rejects_blank_array_item(self):
        with self.assertRaisesRegex(PersonaValidationError, "非空字符串"):
            validate_persona(persona(personality=["  "]))

    def test_rejects_missing_required_field(self):
        data = persona()
        del data["facts"]
        with self.assertRaisesRegex(PersonaValidationError, "缺少必填字段"):
            validate_persona(data)

    def test_rejects_invalid_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "persona.json"
            path.write_text('{"name": ', encoding="utf-8")
            with self.assertRaisesRegex(PersonaValidationError, "不是合法 JSON"):
                load_persona(path)


if __name__ == "__main__":
    unittest.main()
