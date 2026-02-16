import pytest
from pydantic import ValidationError

from bilancio.config.models import TransferClaim


def test_transfer_claim_requires_reference():
    with pytest.raises(ValidationError):
        TransferClaim(to_agent="F3")


def test_transfer_claim_accepts_alias_or_id():
    # alias only
    m1 = TransferClaim(contract_alias="ALIAS1", to_agent="F3")
    assert m1.contract_alias == "ALIAS1"
    assert m1.contract_id is None
    # id only
    m2 = TransferClaim(contract_id="C_123", to_agent="F3")
    assert m2.contract_id == "C_123"


def test_transfer_claim_alias_and_id_model_allows_both():
    # Model permits both present; apply layer enforces equality.
    m = TransferClaim(contract_alias="ALIASX", contract_id="C_YYY", to_agent="F3")
    assert m.contract_alias == "ALIASX"
    assert m.contract_id == "C_YYY"
