from pathlib import Path

def get_project_directory_path():
    # Get the current file's absolute path.
    current_filepath = Path(__file__).resolve()

    # Assume the directory structure has not changed.
    number_of_parents_to_project_path = 1

    return current_filepath.parents[number_of_parents_to_project_path]

def get_tests_directory_path():
    return get_project_directory_path() / "tests"

def get_data_directory_path():
    return get_tests_directory_path() / "Data"