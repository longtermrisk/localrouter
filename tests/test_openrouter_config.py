"""Tests for OpenRouter provider preferences config system."""

import os
import tempfile
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from localrouter.llm import (
    _DEFAULT_OPENROUTER_PROVIDERS,
    _deep_merge,
    _load_config,
    _get_openrouter_provider_prefs,
    get_response_factory,
)


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {
            "openrouter": {
                "providers": {"qwen": {"only": ["alibaba"], "allow_fallbacks": False}}
            }
        }
        override = {
            "openrouter": {"providers": {"qwen": {"only": ["alibaba", "deepinfra"]}}}
        }
        result = _deep_merge(base, override)
        assert result["openrouter"]["providers"]["qwen"]["only"] == [
            "alibaba",
            "deepinfra",
        ]
        assert result["openrouter"]["providers"]["qwen"]["allow_fallbacks"] is False

    def test_add_new_key(self):
        base = {"openrouter": {"providers": {"qwen": {"only": ["alibaba"]}}}}
        override = {"openrouter": {"providers": {"meta-llama": {"only": ["meta"]}}}}
        result = _deep_merge(base, override)
        assert "qwen" in result["openrouter"]["providers"]
        assert "meta-llama" in result["openrouter"]["providers"]

    def test_base_unchanged(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1  # original not mutated


class TestDefaultProviders:
    def test_qwen_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["qwen"]["only"] == ["alibaba"]
        assert _DEFAULT_OPENROUTER_PROVIDERS["qwen"]["allow_fallbacks"] is False
        assert _DEFAULT_OPENROUTER_PROVIDERS["qwen"]["require_parameters"] is True

    def test_deepseek_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["deepseek"]["only"] == ["deepseek"]

    def test_minimax_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["minimax"]["only"] == ["minimax"]

    def test_xai_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["x-ai"]["only"] == ["xai"]

    def test_moonshotai_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["moonshotai"]["only"] == ["moonshotai"]

    def test_mistralai_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["mistralai"]["only"] == ["mistral"]


class TestGetOpenrouterProviderPrefs:
    def test_known_model(self):
        prefs = _get_openrouter_provider_prefs("qwen/qwen3-coder")
        assert prefs is not None
        assert prefs["only"] == ["alibaba"]

    def test_deepseek_model(self):
        prefs = _get_openrouter_provider_prefs("deepseek/deepseek-v3.2")
        assert prefs is not None
        assert prefs["only"] == ["deepseek"]

    def test_xai_model(self):
        prefs = _get_openrouter_provider_prefs("x-ai/grok-4")
        assert prefs is not None
        assert prefs["only"] == ["xai"]

    def test_unknown_org_returns_none(self):
        prefs = _get_openrouter_provider_prefs("meta-llama/llama-3.3-70b")
        assert prefs is None

    def test_no_slash_returns_none(self):
        prefs = _get_openrouter_provider_prefs("gpt-5")
        assert prefs is None


class TestLoadConfig:
    def test_loads_defaults_when_no_files(self):
        with patch("os.path.exists", return_value=False):
            config = _load_config()
        assert "qwen" in config["openrouter"]["providers"]
        assert "deepseek" in config["openrouter"]["providers"]

    def test_global_config_overrides_defaults(self):
        yaml_content = """
openrouter:
  providers:
    qwen:
      only: ["alibaba", "deepinfra"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        def exists_side_effect(path):
            if "/.localrouter.yaml" in path and "~" not in path:
                return path == temp_path
            return path == temp_path

        try:
            with (
                patch("os.path.expanduser", return_value=temp_path),
                patch("os.path.exists", side_effect=lambda p: p == temp_path or False),
                patch("os.getcwd", return_value="/nonexistent"),
            ):
                config = _load_config()
            assert config["openrouter"]["providers"]["qwen"]["only"] == [
                "alibaba",
                "deepinfra",
            ]
            # Other defaults still present
            assert config["openrouter"]["providers"]["deepseek"]["only"] == ["deepseek"]
        finally:
            os.unlink(temp_path)

    def test_local_overrides_global(self):
        global_yaml = """
openrouter:
  providers:
    qwen:
      only: ["alibaba"]
"""
        local_yaml = """
openrouter:
  providers:
    qwen:
      only: ["deepinfra"]
"""
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as gf,
            tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as lf,
        ):
            gf.write(global_yaml)
            gf.flush()
            lf.write(local_yaml)
            lf.flush()
            global_path = gf.name
            local_path = lf.name

        try:
            with (
                patch("os.path.expanduser", return_value=global_path),
                patch("os.path.exists", return_value=True),
                patch("os.path.join", return_value=local_path),
                patch("os.getcwd", return_value="/tmp"),
            ):
                config = _load_config()
            assert config["openrouter"]["providers"]["qwen"]["only"] == ["deepinfra"]
        finally:
            os.unlink(global_path)
            os.unlink(local_path)


class TestExtraBodyInjection:
    """Test that get_response_factory injects extra_body when extra_body_fn is provided."""

    @pytest.mark.asyncio
    async def test_extra_body_injected(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock(
            role="assistant",
            content="hello",
            tool_calls=None,
            refusal=None,
        )

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        def my_extra_body_fn(model):
            if model.startswith("qwen/"):
                return {"provider": {"only": ["alibaba"], "allow_fallbacks": False}}
            return None

        factory_fn = get_response_factory(mock_client, extra_body_fn=my_extra_body_fn)

        from localrouter import ChatMessage, MessageRole, TextBlock

        messages = [ChatMessage(role=MessageRole.user, content=[TextBlock(text="hi")])]

        await factory_fn(messages=messages, tools=None, model="qwen/qwen3-coder")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert "extra_body" in call_kwargs.kwargs or "extra_body" in (
            call_kwargs[1] if len(call_kwargs) > 1 else {}
        )
        # Check the actual kwargs passed
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert all_kwargs["extra_body"]["provider"]["only"] == ["alibaba"]

    @pytest.mark.asyncio
    async def test_no_extra_body_for_unknown_model(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock(
            role="assistant",
            content="hello",
            tool_calls=None,
            refusal=None,
        )

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        def my_extra_body_fn(model):
            return None  # No preferences

        factory_fn = get_response_factory(mock_client, extra_body_fn=my_extra_body_fn)

        from localrouter import ChatMessage, MessageRole, TextBlock

        messages = [ChatMessage(role=MessageRole.user, content=[TextBlock(text="hi")])]

        await factory_fn(
            messages=messages, tools=None, model="meta-llama/llama-3.3-70b"
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "extra_body" not in all_kwargs

    @pytest.mark.asyncio
    async def test_no_extra_body_fn(self):
        """When no extra_body_fn is provided, no extra_body should be added."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock(
            role="assistant",
            content="hello",
            tool_calls=None,
            refusal=None,
        )

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        factory_fn = get_response_factory(mock_client)  # No extra_body_fn

        from localrouter import ChatMessage, MessageRole, TextBlock

        messages = [ChatMessage(role=MessageRole.user, content=[TextBlock(text="hi")])]

        await factory_fn(messages=messages, tools=None, model="gpt-5")

        call_kwargs = mock_client.chat.completions.create.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "extra_body" not in all_kwargs
