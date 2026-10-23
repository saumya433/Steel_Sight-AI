from steelsight.detector import Detection
from steelsight.gate import Decision, GateConfig, decide


def det(name, conf, box=(10, 10, 50, 50)):
    return Detection(0, name, conf, box)


def test_pass_when_nothing_above_floor():
    cfg = GateConfig(review_min=0.25, auto_flag=0.6)
    r = decide([det("scratches", 0.1)], cfg)
    assert r.decision == Decision.PASS and not r.kept


def test_review_between_floors():
    cfg = GateConfig(review_min=0.25, auto_flag=0.6)
    r = decide([det("scratches", 0.4)], cfg)
    assert r.decision == Decision.REVIEW and len(r.kept) == 1 and not r.auto


def test_auto_flag_above_auto_floor():
    cfg = GateConfig(review_min=0.25, auto_flag=0.6)
    r = decide([det("patches", 0.9), det("scratches", 0.3)], cfg)
    assert r.decision == Decision.AUTO_FLAG and len(r.kept) == 2 and len(r.auto) == 1


def test_per_class_floors_override_defaults():
    cfg = GateConfig(review_min=0.25, auto_flag=0.6, per_class_review={"crazing": 0.15}, per_class_auto={"patches": 0.8})
    assert decide([det("crazing", 0.2)], cfg).decision == Decision.REVIEW      # below default, above class floor
    assert decide([det("patches", 0.7)], cfg).decision == Decision.REVIEW      # above default auto, below class auto


def test_unknown_when_only_stage1_fires():
    cfg = GateConfig(anomaly_threshold=1.0)
    r = decide([], cfg, anomaly_score=1.4)
    assert r.decision == Decision.UNKNOWN and r.anomaly_fired


def test_stage1_does_not_downgrade_a_detection():
    cfg = GateConfig(anomaly_threshold=1.0)
    r = decide([det("inclusion", 0.5)], cfg, anomaly_score=1.4)
    assert r.decision == Decision.REVIEW and "anomaly" in r.reason


def test_min_box_fraction_filters_specks():
    cfg = GateConfig(min_box_frac=0.01)
    r = decide([det("inclusion", 0.9, (0, 0, 5, 5))], cfg, image_hw=(200, 200))
    assert r.decision == Decision.PASS


def test_config_roundtrip(tmp_path):
    cfg = GateConfig(review_min=0.3, per_class_review={"crazing": 0.18}, anomaly_threshold=2.5)
    p = tmp_path / "t.json"
    cfg.save(p)
    back = GateConfig.load(p)
    assert back.review_min == 0.3 and back.per_class_review["crazing"] == 0.18 and back.anomaly_threshold == 2.5
