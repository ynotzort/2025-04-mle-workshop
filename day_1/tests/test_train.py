from datetime import date
import tempfile
from duration_prediction.train import train
from pathlib import Path


def test_training_run_regression():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "model.bin"
        mse = train(date(2022, 1, 1), date(2022, 2, 1), out_path=out_path)
        
        assert abs(mse - 8.189) < 0.01


def test_training_creates_a_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "model.bin"
        assert not out_path.exists()
        _ = train(date(2022, 1, 1), date(2022, 2, 1), out_path=out_path)
        assert out_path.exists()
        
