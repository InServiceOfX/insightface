from insightface.utils import ensure_available

from ...testing_utilities import (
	get_data_directory_path,
	get_tests_directory_path)

import os
import pytest

def test_get_data_directory_path():
	data_path = get_data_directory_path()
	assert data_path.is_dir()
	assert "Data" in str(data_path)
	assert "tests" in str(data_path)
	assert "insightface" in str(data_path)
	assert data_path.exists()

def test_download_steps_on_local_directory():
	data_path = get_data_directory_path()
	expanded_data_path = os.path.expanduser(data_path)
	assert str(data_path) == expanded_data_path

	tests_path = get_tests_directory_path()
	expanded_tests_path = os.path.expanduser(tests_path)
	assert str(tests_path) == expanded_tests_path

	name="antelopev2"
	dir_path = os.path.join(str(tests_path), "Data", name)

	assert os.path.exists(dir_path)

def test_ensure_available_returns_existing_local_directory():
	tests_path = get_tests_directory_path()
	name="antelopev2"
	dir_path = ensure_available("Data", name, str(tests_path))
	assert dir_path == str(tests_path / "Data" / name)
