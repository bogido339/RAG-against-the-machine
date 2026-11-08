
from pathlib import Path
from unittest.mock import patch

import pytest

from src.indexer import Indexer


def test_mkdir_permission_denied():
    indexer = Indexer()

    with patch.object(
        Path,
        "mkdir",
        side_effect=PermissionError("Permission denied: data/processed")
    ):
        with pytest.raises(PermissionError):
            indexer._save_chunks([])
