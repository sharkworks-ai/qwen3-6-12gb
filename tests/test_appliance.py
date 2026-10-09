from pathlib import Path

from appliance.core import JOBS, ApplianceDB, peak_vram
from appliance.publish import inside


def test_db_round_trip(tmp_path: Path):
    db = ApplianceDB(tmp_path / "db.sqlite3")
    db.create_job("r1", "smoke", {"a": 1})
    db.update_job("r1", status="running", pid=123)
    job = db.get_job("r1")
    assert job.status == "running"
    assert job.pid == 123
    assert job.config == {"a": 1}


def test_registered_job():
    assert JOBS["smoke"].gpu_required is True


def test_publish_path_guard(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    child = data / "artifacts"
    child.mkdir()
    assert inside(data, child) == child.resolve()


def test_publish_path_guard_rejects_escape(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        inside(data, outside)
    except ValueError:
        pass
    else:
        raise AssertionError("expected path escape rejection")


def test_peak_vram(tmp_path: Path):
    path = tmp_path / "vram.csv"
    path.write_text(
        "timestamp,index,name,memory_total_mib,memory_used_mib,utilization_gpu_percent,temperature_c,power_w\n"
        "t,0,GPU,32768,100,1,40,20\n"
        "t,0,GPU,32768,300,2,41,25\n"
        "t,1,GPU,32768,250,2,41,25\n",
        encoding="utf-8",
    )
    assert peak_vram(path) == {"0": 300.0, "1": 250.0}
