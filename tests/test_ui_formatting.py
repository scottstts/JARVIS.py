"""Unit tests for Telegram HTML rendering."""

from __future__ import annotations

import unittest

from ui.telegram.formatting import render_markdown_to_telegram_html


class TelegramFormattingTests(unittest.TestCase):
    def test_renders_basic_inline_markdown(self) -> None:
        rendered = render_markdown_to_telegram_html(
            "**bold** *italic* ~~gone~~ ||secret|| `code`"
        )
        self.assertEqual(
            rendered,
            "<b>bold</b> <i>italic</i> <s>gone</s> <tg-spoiler>secret</tg-spoiler> <code>code</code>",
        )

    def test_renders_links_headings_and_blockquotes(self) -> None:
        rendered = render_markdown_to_telegram_html(
            "# Heading\n> quoted line\n[OpenAI](https://openai.com)"
        )
        self.assertEqual(
            rendered,
            '<b>Heading</b>\n<blockquote>quoted line</blockquote>\n<a href="https://openai.com">OpenAI</a>',
        )

    def test_renders_fenced_code_blocks(self) -> None:
        rendered = render_markdown_to_telegram_html("```python\nprint('hi')\n```")
        self.assertEqual(
            rendered,
            '<pre><code class="language-python">print(&#x27;hi&#x27;)</code></pre>',
        )

    def test_unmatched_markers_remain_literal(self) -> None:
        rendered = render_markdown_to_telegram_html("**open and [broken](not-a-url)")
        self.assertEqual(rendered, "**open and [broken](not-a-url)")
