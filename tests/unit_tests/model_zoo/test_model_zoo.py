from insightface.model_zoo import model_zoo
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.model_zoo.attribute import Attribute
from insightface.model_zoo.landmark import Landmark
from insightface.model_zoo.retinaface import RetinaFace

from ...testing_utilities import (
	get_data_directory_path
	)

import glob, os, pytest

class TestSetup:
	__test__ = False
	def __init__(self):
		self.name = "antelopev2"
		self.data_path = get_data_directory_path()
		self.model_dir = self.data_path / self.name

		onnx_files = glob.glob(os.path.join(str(self.model_dir), '*.onnx'))
		self.onnx_files = sorted(onnx_files)


def test_get_model_steps_explicitly():

	name = "antelopev2"
	data_path = get_data_directory_path()
	model_dir = data_path / name
	assert model_dir.exists()

	onnx_files = glob.glob(os.path.join(str(model_dir), '*.onnx'))
	onnx_files = sorted(onnx_files)

	assert len(onnx_files) == 5

	name0 = onnx_files[0]

	assert name0.endswith('.onnx')

	model_file = name0
	assert os.path.exists(model_file)

	# Because the above 2 assertions are true, we skip over all the conditional
	# statements before constructing a model router.

def test_ModelRouter_constructs():
	test_setup = TestSetup()

	router = model_zoo.ModelRouter(test_setup.onnx_files[0])

	assert router.onnx_file == test_setup.onnx_files[0]

def test_ModelRouter_get_model_returns():
	test_setup = TestSetup()
	router = model_zoo.ModelRouter(test_setup.onnx_files[0])
	kwargs = {"providers": ['CUDAExecutionProvider',],
		"enable_cuda_graph": 1}

	model = router.get_model(**kwargs)

	assert type(model) == Landmark
	assert model.taskname == "landmark_3d_68"

	router = model_zoo.ModelRouter(test_setup.onnx_files[1])
	model = router.get_model(**kwargs)

	assert type(model) == Landmark
	assert model.taskname == "landmark_2d_106"

	router = model_zoo.ModelRouter(test_setup.onnx_files[2])
	model = router.get_model(**kwargs)

	assert type(model) == Attribute
	assert model.taskname == "genderage"

	router = model_zoo.ModelRouter(test_setup.onnx_files[3])
	model = router.get_model(**kwargs)

	assert type(model) == ArcFaceONNX
	assert model.taskname == "recognition"

	router = model_zoo.ModelRouter(test_setup.onnx_files[4])
	model = router.get_model(**kwargs)

	assert type(model) == RetinaFace
	assert model.taskname == "detection"
