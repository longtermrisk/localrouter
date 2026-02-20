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
                "providers": {
                    "qwen": {"order": ["alibaba"], "require_parameters": True}
                }
            }
        }
        override = {
            "openrouter": {"providers": {"qwen": {"order": ["alibaba", "deepinfra"]}}}
        }
        result = _deep_merge(base, override)
        assert result["openrouter"]["providers"]["qwen"]["order"] == [
            "alibaba",
            "deepinfra",
        ]
        assert result["openrouter"]["providers"]["qwen"]["require_parameters"] is True

    def test_add_new_key(self):
        base = {"openrouter": {"providers": {"qwen": {"order": ["alibaba"]}}}}
        override = {"openrouter": {"providers": {"meta-llama": {"order": ["meta"]}}}}
        result = _deep_merge(base, override)
        assert "qwen" in result["openrouter"]["providers"]
        assert "meta-llama" in result["openrouter"]["providers"]

    def test_base_unchanged(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1  # original not mutated


class TestDefaultProviders:
    """Verify built-in defaults use 'order' for orgs with official providers
    and just 'require_parameters' for orgs without."""

    def test_qwen_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["qwen"]["order"] == ["alibaba"]
        assert _DEFAULT_OPENROUTER_PROVIDERS["qwen"]["require_parameters"] is True
        assert "only" not in _DEFAULT_OPENROUTER_PROVIDERS["qwen"]

    def test_deepseek_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["deepseek"]["order"] == ["deepseek"]
        assert _DEFAULT_OPENROUTER_PROVIDERS["deepseek"]["require_parameters"] is True

    def test_xai_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["x-ai"]["order"] == ["xai"]
        assert _DEFAULT_OPENROUTER_PROVIDERS["x-ai"]["require_parameters"] is True

    def test_mistralai_defaults(self):
        assert _DEFAULT_OPENROUTER_PROVIDERS["mistralai"]["order"] == ["mistral"]
        assert _DEFAULT_OPENROUTER_PROVIDERS["mistralai"]["require_parameters"] is True

    def test_moonshotai_defaults_no_official_provider(self):
        assert "order" not in _DEFAULT_OPENROUTER_PROVIDERS["moonshotai"]
        assert "only" not in _DEFAULT_OPENROUTER_PROVIDERS["moonshotai"]
        assert _DEFAULT_OPENROUTER_PROVIDERS["moonshotai"]["require_parameters"] is True

    def test_minimax_defaults_no_official_provider(self):
        assert "order" not in _DEFAULT_OPENROUTER_PROVIDERS["minimax"]
        assert "only" not in _DEFAULT_OPENROUTER_PROVIDERS["minimax"]
        assert _DEFAULT_OPENROUTER_PROVIDERS["minimax"]["require_parameters"] is True

    def test_zai_defaults_no_official_provider(self):
        assert "order" not in _DEFAULT_OPENROUTER_PROVIDERS["z-ai"]
        assert "only" not in _DEFAULT_OPENROUTER_PROVIDERS["z-ai"]
        assert _DEFAULT_OPENROUTER_PROVIDERS["z-ai"]["require_parameters"] is True


class TestGetOpenrouterProviderPrefs:
    def test_qwen_model(self):
        prefs = _get_openrouter_provider_prefs("qwen/qwen3-coder")
        assert prefs is not None
        assert prefs["order"] == ["alibaba"]
        assert prefs["require_parameters"] is True

    def test_deepseek_model(self):
        prefs = _get_openrouter_provider_prefs("deepseek/deepseek-v3.2")
        assert prefs is not None
        assert prefs["order"] == ["deepseek"]

    def test_xai_model(self):
        prefs = _get_openrouter_provider_prefs("x-ai/grok-4")
        assert prefs is not None
        assert prefs["order"] == ["xai"]

    def test_moonshotai_kimi_k2_5(self):
        """Regression: moonshotai/kimi-k2.5 must not 404."""
        prefs = _get_openrouter_provider_prefs("moonshotai/kimi-k2.5")
        assert prefs is not None
        assert "only" not in prefs
        assert prefs["require_parameters"] is True

    def test_minimax_m2_5(self):
        """Regression: minimax/minimax-m2.5 must not 404."""
        prefs = _get_openrouter_provider_prefs("minimax/minimax-m2.5")
        assert prefs is not None
        assert "only" not in prefs
        assert prefs["require_parameters"] is True

    def test_zai_glm5(self):
        """Regression: z-ai/glm-5 must not 404."""
        prefs = _get_openrouter_provider_prefs("z-ai/glm-5")
        assert prefs is not None
        assert "only" not in prefs
        assert prefs["require_parameters"] is True

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
        assert "moonshotai" in config["openrouter"]["providers"]
        assert "minimax" in config["openrouter"]["providers"]
        assert "z-ai" in config["openrouter"]["providers"]

    def test_global_config_overrides_defaults(self):
        yaml_content = """
openrouter:
  providers:
    qwen:
      order: ["alibaba", "deepinfra"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            with (
                patch("os.path.expanduser", return_value=temp_path),
                patch("os.path.exists", side_effect=lambda p: p == temp_path or False),
                patch("os.getcwd", return_value="/nonexistent"),
            ):
                config = _load_config()
            assert config["openrouter"]["providers"]["qwen"]["order"] == [
                "alibaba",
                "deepinfra",
            ]
            # Other defaults still present
            assert config["openrouter"]["providers"]["deepseek"]["order"] == [
                "deepseek"
            ]
        finally:
            os.unlink(temp_path)

    def test_local_overrides_global(self):
        global_yaml = """
openrouter:
  providers:
    qwen:
      order: ["alibaba"]
"""
        local_yaml = """
openrouter:
  providers:
    qwen:
      order: ["deepinfra"]
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
            assert config["openrouter"]["providers"]["qwen"]["order"] == ["deepinfra"]
        finally:
            os.unlink(global_path)
            os.unlink(local_path)


class TestExtraBodyInjection:
    """Test that get_response_factory injects extra_body when extra_body_fn is provided."""

    @pytest.mark.asyncio
    async def test_extra_body_injected_with_order(self):
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
                return {
                    "provider": {
                        "order": ["alibaba"],
                        "require_parameters": True,
                    }
                }
            return None

        factory_fn = get_response_factory(mock_client, extra_body_fn=my_extra_body_fn)

        from localrouter import ChatMessage, MessageRole, TextBlock

        messages = [ChatMessage(role=MessageRole.user, content=[TextBlock(text="hi")])]

        await factory_fn(messages=messages, tools=None, model="qwen/qwen3-coder")

        call_kwargs = mock_client.chat.completions.create.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert all_kwargs["extra_body"]["provider"]["order"] == ["alibaba"]
        assert all_kwargs["extra_body"]["provider"]["require_parameters"] is True

    @pytest.mark.asyncio
    async def test_extra_body_injected_require_parameters_only(self):
        """For orgs without official providers, only require_parameters is set."""
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
            return {"provider": {"require_parameters": True}}

        factory_fn = get_response_factory(mock_client, extra_body_fn=my_extra_body_fn)

        from localrouter import ChatMessage, MessageRole, TextBlock

        messages = [ChatMessage(role=MessageRole.user, content=[TextBlock(text="hi")])]

        await factory_fn(messages=messages, tools=None, model="moonshotai/kimi-k2.5")

        call_kwargs = mock_client.chat.completions.create.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert all_kwargs["extra_body"]["provider"]["require_parameters"] is True
        assert "only" not in all_kwargs["extra_body"]["provider"]

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
