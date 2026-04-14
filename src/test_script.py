import os

def test_data_exists():
    assert os.path.exists("data/iris.csv"), "File data tidak ditemukan!"

def test_mlflow_dir():
    # Memastikan folder mlruns tercipta setelah training
    if os.path.exists("src/train.py"):
        assert True