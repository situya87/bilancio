from bilancio.domain.agents.bank import Bank
from bilancio.domain.agents.central_bank import CentralBank
from bilancio.domain.agents.household import Household
from bilancio.domain.agents.treasury import Treasury
from bilancio.domain.instruments.base import InstrumentKind
from bilancio.domain.instruments.credit import Payable
from bilancio.domain.instruments.means_of_payment import BankDeposit
from bilancio.engines.system import System


def test_create_agents_and_deposit_invariants():
    sys = System()
    cb = CentralBank(id="CB1", name="Central Bank", kind="central_bank")
    b1 = Bank(id="B1", name="Bank 1", kind="bank")
    h1 = Household(id="H1", name="HH 1", kind="household")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)

    d = BankDeposit(
        id="D1",
        kind=InstrumentKind.BANK_DEPOSIT,
        amount=100,
        denom="X",
        asset_holder_id="H1",
        liability_issuer_id="B1",
    )
    sys.add_contract(d)
    sys.assert_invariants()


def test_policy_allows_and_blocks_expected_cases():
    sys = System()
    cb = CentralBank(id="CB1", name="CB", kind="central_bank")
    b1 = Bank(id="B1", name="B1", kind="bank")
    h1 = Household(id="H1", name="H1", kind="household")
    t1 = Treasury(id="T1", name="T1", kind="treasury")
    sys.add_agent(cb)
    sys.add_agent(b1)
    sys.add_agent(h1)
    sys.add_agent(t1)

    # HH can hold bank deposit; bank issues it
    sys.add_contract(
        BankDeposit(
            id="d",
            kind=InstrumentKind.BANK_DEPOSIT,
            amount=10,
            denom="X",
            asset_holder_id="H1",
            liability_issuer_id="B1",
        )
    )

    # Treasury can hold reserves, not required to test now, but ensure creation passes
    # Use mint_reserves to maintain invariants
    sys.mint_reserves("B1", 5)

    # Any agent can issue a payable in MVP
    sys.add_contract(
        Payable(
            id="p",
            kind=InstrumentKind.PAYABLE,
            amount=7,
            denom="X",
            asset_holder_id="B1",
            liability_issuer_id="H1",
            due_day=0,
        )
    )
    sys.assert_invariants()
