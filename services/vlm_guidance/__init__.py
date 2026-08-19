"""Remote VLM guidance service."""

SKILL_NAMES = ("push", "pick", "place", "insert", "screw")
# Version 2 is the legacy dual-head checkpoint. Version 3 is native/original
# text SFT, where skill and the raw-pixel point are parsed from generated JSON.
POINT_POLICY_VERSION = 2
ORIGINAL_SFT_POLICY_VERSION = 3
