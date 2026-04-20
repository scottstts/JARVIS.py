"""Unit tests for request token estimation heuristics."""

from __future__ import annotations

import unittest

from jarvis.core.token_estimator import estimate_request_input_tokens
from jarvis.llm import ImagePart, LLMMessage, LLMRequest, ToolCall


class TokenEstimatorTests(unittest.TestCase):
    def test_base64_payload_size_does_not_dominate_image_estimation(self) -> None:
        small_request = LLMRequest(
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart.from_base64(
                            media_type="image/png",
                            data_base64="AAAA",
                            detail="auto",
                        ),
                    ),
                ),
            ),
        )
        large_request = LLMRequest(
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart.from_base64(
                            media_type="image/png",
                            data_base64="A" * 500_000,
                            detail="auto",
                        ),
                    ),
                ),
            ),
        )

        self.assertEqual(
            estimate_request_input_tokens(small_request),
            estimate_request_input_tokens(large_request),
        )

    def test_high_detail_images_cost_more_than_auto_detail_images(self) -> None:
        auto_request = LLMRequest(
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart.from_base64(
                            media_type="image/png",
                            data_base64="AAAA",
                            detail="auto",
                        ),
                    ),
                ),
            ),
        )
        high_request = LLMRequest(
            messages=(
                LLMMessage(
                    role="user",
                    parts=(
                        ImagePart.from_base64(
                            media_type="image/png",
                            data_base64="AAAA",
                            detail="high",
                        ),
                    ),
                ),
            ),
        )

        self.assertEqual(
            estimate_request_input_tokens(high_request)
            - estimate_request_input_tokens(auto_request),
            128,
        )

    def test_tool_call_estimation_uses_raw_arguments_size(self) -> None:
        small_request = LLMRequest(
            messages=(
                LLMMessage(
                    role="assistant",
                    parts=(
                        ToolCall(
                            call_id="call_1",
                            name="file_patch",
                            arguments={"operation": "write"},
                            raw_arguments='{"operation":"write"}',
                        ),
                    ),
                ),
            ),
        )
        large_request = LLMRequest(
            messages=(
                LLMMessage(
                    role="assistant",
                    parts=(
                        ToolCall(
                            call_id="call_1",
                            name="file_patch",
                            arguments={"operation": "write"},
                            raw_arguments='{"operation":"write","content":"' + ("A" * 4000) + '"}',
                        ),
                    ),
                ),
            ),
        )

        self.assertGreater(
            estimate_request_input_tokens(large_request),
            estimate_request_input_tokens(small_request),
        )
