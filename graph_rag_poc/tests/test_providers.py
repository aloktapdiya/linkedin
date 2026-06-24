import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
from config import Config
from src.generation.providers.factory import is_ollama_running
from src.generation.providers.ollama import OllamaProvider
from src.generation.generator import Generator


@pytest.fixture
def config():
    return Config()


def test_is_ollama_running_returns_bool(config):
    result = is_ollama_running(config.ollama_base_url, timeout=1.0)
    assert isinstance(result, bool)


def test_generator_provider_is_lazy(config):
    gen = Generator(config)
    assert gen._provider is None


def test_generator_calls_provider_generate(config):
    gen = Generator(config)
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "Paris"
    mock_provider.name = "mock/test"
    with patch("src.generation.generator.create_provider", return_value=mock_provider):
        result = gen.generate("What is the capital of France?", [])
    assert result == "Paris"
    mock_provider.generate.assert_called_once()


def test_ollama_provider_name(config):
    provider = OllamaProvider(config)
    assert provider.name.startswith("ollama/")


def test_factory_falls_back_to_claude_when_no_ollama_no_gemini():
    from src.generation.providers.factory import create_provider
    cfg = Config()
    cfg.gemini_api_key = ""
    cfg.anthropic_api_key = "sk-ant-test-key"
    with patch("src.generation.providers.factory.is_ollama_running", return_value=False), \
         patch("src.generation.providers.claude.anthropic.Anthropic"):
        provider = create_provider(cfg)
    assert "claude" in provider.name


def test_factory_raises_when_no_engine_available():
    from src.generation.providers.factory import create_provider
    cfg = Config()
    cfg.gemini_api_key = ""
    cfg.anthropic_api_key = ""
    with patch("src.generation.providers.factory.is_ollama_running", return_value=False):
        with pytest.raises(RuntimeError, match="No inference engine"):
            create_provider(cfg)
