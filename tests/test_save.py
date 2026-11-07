"""save_fdn / load_fdn / resave_fdn: DSS as readable JSON, anything else pickled."""

from __future__ import annotations

import importlib
import json
import pickle
import sys

import numpy as np
import pytest

import pyFDN
from pyFDN import save as save_module


def _impulse(n: int = 2048) -> np.ndarray:
    x = np.zeros(n)
    x[0] = 1.0
    return x


def test_a_dss_is_saved_as_readable_json_and_round_trips(tmp_path) -> None:
    dss = pyFDN.get_fdn_preset("colorless_N4_d1")
    path = tmp_path / "fdn.json"
    pyFDN.save_fdn(dss, path)

    data = json.loads(path.read_text())
    assert data["format"] == "pyfdn-dss"
    assert data["pyfdn_version"] == pyFDN.__version__
    assert '"delays": [1499, 1889, 2381, 2999]' in path.read_text()  # one line

    restored = pyFDN.load_fdn(path)
    assert isinstance(restored, pyFDN.DSS)
    for got, want in zip(restored, dss, strict=True):
        np.testing.assert_array_equal(got, want)  # floats round-trip exactly


def test_a_fir_feedback_matrix_saves_too(tmp_path) -> None:
    A, _ = pyFDN.construct_cascaded_paraunitary_matrix(4, 2, matrix_type="random")
    dss = pyFDN.DSS(
        np.array([19, 31, 42, 57]),
        A,
        np.ones((4, 1)),
        np.ones((1, 4)),
        np.zeros((1, 1)),
    )
    pyFDN.save_fdn(dss, tmp_path / "fir.json")
    np.testing.assert_array_equal(pyFDN.load_fdn(tmp_path / "fir.json").A, A)


def test_a_json_file_that_is_not_a_dss_is_refused(tmp_path) -> None:
    (tmp_path / "other.json").write_text('{"format": "something-else"}')
    with pytest.raises(ValueError, match="not a pyfdn-dss file"):
        pyFDN.load_fdn(tmp_path / "other.json")


def test_anything_else_is_pickled_with_its_versions(tmp_path) -> None:
    dss, filters = pyFDN.fdn_dss_gallery(4, rt=0.5, rng=3)
    pyFDN.save_fdn((dss, filters), tmp_path / "fdn.pkl")
    with open(tmp_path / "fdn.pkl", "rb") as file:
        header = pickle.load(file)
    assert header["pyfdn_version"] == pyFDN.__version__
    assert header["python"].count(".") == 2
    restored, restored_filters = pyFDN.load_fdn(tmp_path / "fdn.pkl")
    np.testing.assert_array_equal(restored.A, dss.A)
    np.testing.assert_array_equal(restored_filters["post_delay"], filters["post_delay"])


def test_a_td_graph_round_trips_with_its_state(tmp_path) -> None:
    dss, filters = pyFDN.fdn_dss_gallery(4, rt=0.5, rng=3)
    graph = pyFDN.dss_to_td(*dss, **filters)
    graph.process_signal(_impulse(100))  # leave the delay lines non-empty
    pyFDN.save_fdn(graph, tmp_path / "graph.pkl")

    restored = pyFDN.load_fdn(tmp_path / "graph.pkl")
    np.testing.assert_array_equal(
        restored.process_signal(np.zeros(500)), graph.process_signal(np.zeros(500))
    )


def test_a_flamo_model_round_trips(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("flamo")
    dss, filters = pyFDN.fdn_dss_gallery(4, rt=0.5, rng=3)
    model = pyFDN.trainable_from_dss(dss, 48_000.0, **filters, nfft=2**10)
    pyFDN.save_fdn(model, tmp_path / "model.pkl")

    restored = pyFDN.load_fdn(tmp_path / "model.pkl")
    x = torch.zeros(1, 2**10, 1)
    x[0, 0] = 1.0
    torch.testing.assert_close(restored(x), model(x))


def test_a_moved_class_loads_through_renamed_and_resaves(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "old_home.py").write_text("class Thing:\n    value = 1\n")
    old = importlib.import_module("old_home")
    path = tmp_path / "thing.pkl"
    pyFDN.save_fdn(old.Thing(), path)

    # the refactor: the class moves, the old module is gone
    (tmp_path / "old_home.py").unlink()
    monkeypatch.delitem(sys.modules, "old_home")
    (tmp_path / "new_home.py").write_text("class Thing:\n    value = 2\n")
    with pytest.raises(RuntimeError, match="saved with pyFDN .* on Python"):
        pyFDN.load_fdn(path)

    monkeypatch.setitem(
        save_module.RENAMED, ("old_home", "Thing"), ("new_home", "Thing")
    )
    assert pyFDN.load_fdn(path).value == 2

    pyFDN.resave_fdn(path)
    monkeypatch.delitem(save_module.RENAMED, ("old_home", "Thing"))
    assert pyFDN.load_fdn(path).value == 2
