
from bilancio.core.errors import ValidationError
from bilancio.core.ids import new_id
from bilancio.domain.instruments.base import Instrument


def fungible_key(instr: Instrument) -> tuple:
    # Same type, denomination, issuer, holder → can merge
    return (instr.kind, instr.denom, instr.liability_issuer_id, instr.asset_holder_id)

def is_divisible(instr: Instrument) -> bool:
    # Cash and bank deposits are divisible
    if instr.kind in ("cash", "bank_deposit", "reserve_deposit"):
        return True
    return False

def split(system, instr_id: str, amount: int) -> str:
    instr = system.state.contracts[instr_id]
    if amount <= 0 or amount > instr.amount:
        raise ValidationError("invalid split amount")
    if not is_divisible(instr):
        raise ValidationError("instrument is not divisible")
    # reduce original
    instr.amount -= amount
    # create twin
    twin_id = new_id("C")
    extra_fields = {}
    for field_name in (
        "due_day",
        "maturity_distance",
        "remuneration_rate",
        "issuance_day",
        "last_interest_day",
    ):
        if hasattr(instr, field_name):
            extra_fields[field_name] = getattr(instr, field_name)
    twin = type(instr)(
        id=twin_id,
        kind=instr.kind,
        amount=amount,
        denom=instr.denom,
        asset_holder_id=instr.asset_holder_id,
        liability_issuer_id=instr.liability_issuer_id,
        **extra_fields
    )
    system.add_contract(twin)  # attaches to holder/issuer lists too
    return twin_id

def merge(system, a_id: str, b_id: str) -> str:
    if a_id == b_id:
        return a_id
    a = system.state.contracts[a_id]
    b = system.state.contracts[b_id]
    if fungible_key(a) != fungible_key(b):
        raise ValidationError("instruments are not fungible-compatible")
    a.amount += b.amount
    # detach b from registries
    holder = system.state.agents[b.asset_holder_id]
    issuer = system.state.agents[b.liability_issuer_id]
    holder.asset_ids.remove(b_id)
    issuer.liability_ids.remove(b_id)
    del system.state.contracts[b_id]
    system.log("InstrumentMerged", keep=a_id, removed=b_id)
    return a_id

def consume(system, instr_id: str, amount: int) -> None:
    instr = system.state.contracts[instr_id]
    if amount <= 0 or amount > instr.amount:
        raise ValidationError("invalid consume amount")
    instr.amount -= amount
    if instr.amount == 0:
        holder = system.state.agents[instr.asset_holder_id]
        issuer = system.state.agents[instr.liability_issuer_id]
        holder.asset_ids.remove(instr_id)
        issuer.liability_ids.remove(instr_id)
        del system.state.contracts[instr_id]

def coalesce_deposits(system, customer_id: str, bank_id: str) -> str:
    """Coalesce all deposits for a customer at a bank into a single instrument"""
    ids = system.deposit_ids(customer_id, bank_id)
    if not ids:
        # create a zero-balance deposit instrument
        from bilancio.domain.instruments.means_of_payment import BankDeposit
        dep_id = system.new_contract_id("D")
        dep = BankDeposit(
            id=dep_id, kind="bank_deposit", amount=0, denom="X",
            asset_holder_id=customer_id, liability_issuer_id=bank_id
        )
        system.add_contract(dep)
        return dep_id
    # merge all into the first
    keep = ids[0]
    for other in ids[1:]:
        merge(system, keep, other)
    return keep
