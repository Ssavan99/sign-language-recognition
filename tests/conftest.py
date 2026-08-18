from __future__ import annotations

from pathlib import Path

import pytest

from asl_recognition.data import prepare_manifests
from asl_recognition.smoke import generate_fixture


@pytest.fixture(scope="session")
def prepared_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    root = tmp_path_factory.mktemp("prepared-data")
    source = generate_fixture(root, images_per_class=4, image_size=64, seed=17)
    manifests = root / "manifests"
    result = prepare_manifests(
        source,
        manifests,
        seed=17,
        train_ratio=0.50,
        validation_ratio=0.25,
        test_ratio=0.25,
        near_duplicate_threshold=0,
    )
    return source, manifests, result
