"""
Transforms text chunks into specific training formats.
"""

from __future__ import annotations

from typing import Any, Dict, List

from quarrycore.config.config import FormattingConfig


class Formatter:
    """Applies a specified format to text data for LLM training."""

    def __init__(self, config: FormattingConfig):
        self.config = config

    def format_batch(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Formats a batch of text chunks according to the configured format.

        Args:
            chunks: A list of text chunks to format.

        Returns:
            A list of formatted records.
        """
        if self.config.format_type == "instruction":
            return [self._to_instruction_format(chunk) for chunk in chunks]
        elif self.config.format_type == "document":
            return [{"text": chunk} for chunk in chunks]
        elif self.config.format_type == "conversation":
            return [self._to_conversation_format(chunk) for chunk in chunks]
        else:
            raise ValueError(
                f"Unsupported format type: '{self.config.format_type}'. Supported formats: instruction, document, conversation"
            )

    def _to_instruction_format(self, text: str) -> Dict[str, str]:
        """
        Creates a synthetic instruction-response pair from a text chunk.

        This is a simple implementation. More advanced versions could use another
        LLM to generate more diverse and realistic instructions.
        """
        # The template should contain '{text}' which will be replaced.
        instruction = self.config.instruction_template.format(text="").strip()
        response = text

        # A common format is a dictionary with 'instruction' and 'response' keys.
        # Another popular one is a single 'text' field with roles.
        formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

        return {
            "instruction": instruction,
            "response": response,
            "text": formatted_text,
        }

    def _to_conversation_format(self, text: str) -> Dict[str, Any]:
        """
        Creates a conversation format from a text chunk.

        This creates a simple user-assistant conversation structure.
        """
        # Extract title if possible for the user query
        lines = text.split("\n")
        title = lines[0].strip() if lines else "Tell me about this content"

        # Create conversation format
        return {
            "messages": [
                {"role": "user", "content": f"Can you tell me about: {title}"},
                {"role": "assistant", "content": text},
            ],
            "format": "conversation",
        }
