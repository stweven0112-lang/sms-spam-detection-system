import unittest

from src.text_utils import simple_clean, clean_text


class TestTextUtils(unittest.TestCase):

    def test_non_string_returns_empty(self):
        self.assertEqual(simple_clean(None), "")
        self.assertEqual(simple_clean(123), "")
        self.assertEqual(simple_clean(["spam"]), "")

    def test_lowercase_and_basic_cleanup(self):
        # Lowercase happens before URL substitution; normal words should be lowercase
        self.assertEqual(simple_clean("FREE Prize!!!"), "free prize")

    def test_url_replacement_http(self):
        # NOTE: your implementation substitutes " URL " (uppercase),
        # and does not lower again afterward, so output contains "URL".
        self.assertEqual(simple_clean("Visit http://example.com now!!!"), "visit URL now")

    def test_url_replacement_www(self):
        self.assertEqual(simple_clean("Go to www.example.com now"), "go to URL now")

    def test_remove_non_alnum_and_collapse_spaces(self):
        self.assertEqual(simple_clean("a---b___c"), "a b c")
        self.assertEqual(simple_clean("  hello     world  "), "hello world")

    def test_clean_text_alias_same_as_simple_clean(self):
        text = "Hello!!! www.test.com"
        self.assertEqual(clean_text(text), simple_clean(text))


if __name__ == "__main__":
    unittest.main()