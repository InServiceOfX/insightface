from insightface.app import FaceAnalysis

from ...testing_utilities import (
	get_data_directory_path,
	get_tests_directory_path)

import glob
import onnxruntime
import os
import pytest

class TestSetup:
	__test__ = False
	def __init__(self):
		self.name = "antelopev2"
		self.data_path = get_data_directory_path()
		self.model_dir = self.data_path / self.name

		onnx_files = glob.glob(os.path.join(str(self.model_dir), '*.onnx'))
		self.onnx_files = sorted(onnx_files)

def test_FaceAnalysis_steps_shown():

	name = "antelopev2"
	data_path = get_data_directory_path()
	model_dir = data_path / name
	assert model_dir.exists()

	onnx_files = glob.glob(os.path.join(str(model_dir), '*.onnx'))
	onnx_files = sorted(onnx_files)

	assert len(onnx_files) == 5
	assert "1k3d68.onnx" in onnx_files[0]
	assert "2d106det.onnx" in onnx_files[1]
	assert type(onnx_files[0]) == str

def test_FaceAnalysis_construction():

	test_setup = TestSetup()

	face_analysis = FaceAnalysis(
		name=test_setup.name,
		root=str(test_setup.data_path),
		providers=['CUDAExecutionProvider',])

	assert len(face_analysis.models) == 5
	assert "detection" in face_analysis.models.keys()
	assert "landmark_3d_68" in face_analysis.models.keys()
	assert "landmark_2d_106" in face_analysis.models.keys()