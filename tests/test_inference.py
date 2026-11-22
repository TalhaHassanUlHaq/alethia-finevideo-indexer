from __future__ import annotations

import importlib
import os
import sys
import types


def test_inference_import_sets_offline_defaults(monkeypatch) -> None:
    module_name = "src.althia.inference"
    monkeypatch.delenv("YOLO_OFFLINE", raising=False)
    monkeypatch.delenv("YOLO_AUTOINSTALL", raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    try:
        importlib.import_module(module_name)
        assert os.environ["YOLO_OFFLINE"] == "true"
        assert os.environ["YOLO_AUTOINSTALL"] == "false"
    finally:
        sys.modules.pop(module_name, None)
