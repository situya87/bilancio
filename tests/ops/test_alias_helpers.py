from bilancio.engines.system import System
from bilancio.ops.aliases import get_alias_for_id, get_id_for_alias


def test_alias_helpers_roundtrip():
    sys = System()
    sys.state.aliases["AL1"] = "C_001"
    assert get_id_for_alias(sys, "AL1") == "C_001"
    assert get_alias_for_id(sys, "C_001") == "AL1"
    assert get_id_for_alias(sys, "MISSING") is None
    assert get_alias_for_id(sys, "C_XXX") is None
