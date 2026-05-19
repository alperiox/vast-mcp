from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_state_dir(tmp_path):
    """Provide a temporary state directory instead of ~/.vast-mcp/."""
    return tmp_path


@pytest.fixture
def mock_vast_client():
    """Provide a mock VastAI client."""
    return MagicMock()
