import pathlib
import yaml
import pytest
from pydantic import BaseModel, Field, ValidationError


class DataCfg(BaseModel):
    data_dir: str
    ohlcv_glob: str = "*.parquet"
    min_volume: int = Field(ge=0)


def test_load_data_cfg():
    p = pathlib.Path("config/data.yml")
    assert p.exists(), "config/data.yml missing"
    raw = yaml.safe_load(p.read_text())
    cfg = DataCfg.model_validate(raw)
    assert cfg.data_dir and cfg.ohlcv_glob and cfg.min_volume >= 0


def test_validation_rejects_negative_min_volume():
    with pytest.raises(ValidationError):
        DataCfg.model_validate(
            {"data_dir": "data/raw", "ohlcv_glob": "*.parquet", "min_volume": -1}
        )
