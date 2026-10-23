"""SteelSight AI - two-stage surface defect inspection for steel strip.

Stage 1  anomaly screen   (steelsight.anomaly)   flags anything abnormal, even unseen defects
Stage 2  YOLO detector    (steelsight.detector)  names and localises known defect classes
Gate     decision layer   (steelsight.gate)      PASS / REVIEW / AUTO_FLAG / UNKNOWN
Coil     process layer    (steelsight.coil)      defect map along the coil + roll diagnosis
"""

__version__ = "0.2.0"

from .gate import Decision, GateConfig, decide  # noqa: F401
from .pipeline import InspectionPipeline, InspectionResult  # noqa: F401
