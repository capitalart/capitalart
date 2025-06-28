import json
import os
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
os.environ.setdefault("OPENAI_API_KEY", "test")
import analyze_artwork as aa


def dummy_openai_response(content):
    class Choice:
        def __init__(self, text):
            self.message = type('m', (), {'content': text})
    class Resp:
        def __init__(self, text):
            self.choices = [Choice(text)]
    return Resp(content)


def run_test():
    sample_json = json.dumps({
        "seo_filename": "test-artwork-by-robin-custance-rjc-0001.jpg",
        "title": "Test Artwork – High Resolution Digital Aboriginal Print",
        "description": "Test description " * 50,
        "tags": ["test", "digital art"],
        "materials": ["Digital artwork", "High resolution JPEG file"],
        "primary_colour": "Black",
        "secondary_colour": "Brown"
    })
    with mock.patch.object(aa.client.chat.completions, 'create', return_value=dummy_openai_response(sample_json)):
        system_prompt = aa.read_onboarding_prompt()
        img = next(Path('inputs/artworks').rglob('*.jpg'))
        status = []
        entry = aa.analyze_single(img, system_prompt, None, status)
        print(json.dumps(entry, indent=2)[:200])


if __name__ == '__main__':
    run_test()

