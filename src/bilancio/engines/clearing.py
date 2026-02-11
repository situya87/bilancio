"""Clearing engine (Phase C) for intraday netting and settlement."""

from collections import defaultdict

from bilancio.core.atomic_tx import atomic
from bilancio.core.errors import ValidationError
from bilancio.domain.instruments.credit import Payable


def compute_intraday_nets(system, day: int) -> dict[tuple[str, str], int]:
    """
    Compute net amounts between banks from today's ClientPayment events.
    
    Scans events for ClientPayment events from today and calculates net amounts 
    between each bank pair. Uses lexical ordering for bank pairs (a, b where a < b).
    
    Convention: nets[(a,b)] > 0 means bank a owes bank b
    
    Args:
        system: System instance
        day: Day to compute nets for
        
    Returns:
        Dict mapping bank pairs to net amounts
    """
    nets = defaultdict(int)

    # Scan events for ClientPayment events from today
    for event in system.state.events:
        if event.get("kind") == "ClientPayment" and event.get("day") == day:
            debtor_bank = event.get("payer_bank")
            creditor_bank = event.get("payee_bank")
            amount = event.get("amount", 0)

            if debtor_bank and creditor_bank and debtor_bank != creditor_bank:
                # Use lexical ordering: ensure a < b
                if debtor_bank < creditor_bank:
                    # debtor_bank owes creditor_bank
                    nets[(debtor_bank, creditor_bank)] += amount
                else:
                    # creditor_bank is owed by debtor_bank, so subtract from the reverse pair
                    nets[(creditor_bank, debtor_bank)] -= amount

    return dict(nets)


def settle_intraday_nets(system, day: int):
    """
    Settle intraday nets between banks using reserves or creating overnight payables.
    
    For each net amount between banks:
    - Try to transfer reserves if sufficient
    - If insufficient reserves, create overnight payable due tomorrow
    - Log InterbankCleared or InterbankOvernightCreated events
    
    Args:
        system: System instance  
        day: Current day
    """
    nets = compute_intraday_nets(system, day)

    for (bank_a, bank_b), net_amount in nets.items():
        if net_amount == 0:
            continue

        if net_amount > 0:
            # bank_a owes bank_b
            debtor_bank = bank_a
            creditor_bank = bank_b
            amount = net_amount
        else:
            # bank_b owes bank_a
            debtor_bank = bank_b
            creditor_bank = bank_a
            amount = -net_amount

        # Try to transfer reserves
        try:
            with atomic(system):
                # Find available reserves for debtor bank
                debtor_reserve_ids = []
                for cid in system.state.agents[debtor_bank].asset_ids:
                    contract = system.state.contracts.get(cid)
                    if contract is None:
                        continue
                    if contract.kind == "reserve_deposit":
                        debtor_reserve_ids.append(cid)

                if not debtor_reserve_ids:
                    available_reserves = 0
                else:
                    available_reserves = sum(system.state.contracts[cid].amount for cid in debtor_reserve_ids)

                if available_reserves >= amount:
                    # Sufficient reserves - transfer them
                    system.transfer_reserves(debtor_bank, creditor_bank, amount)
                    system.log("InterbankCleared",
                              debtor_bank=debtor_bank,
                              creditor_bank=creditor_bank,
                              amount=amount)
                else:
                    # Insufficient reserves - create overnight payable
                    payable_id = system.new_contract_id("P")
                    overnight_payable = Payable(
                        id=payable_id,
                        kind="payable",
                        amount=amount,
                        denom="X",
                        asset_holder_id=creditor_bank,
                        liability_issuer_id=debtor_bank,
                        due_day=day + 1
                    )

                    system.add_contract(overnight_payable)
                    system.log("InterbankOvernightCreated",
                              debtor_bank=debtor_bank,
                              creditor_bank=creditor_bank,
                              amount=amount,
                              payable_id=payable_id,
                              due_day=day + 1)

        except ValidationError:
            # If transfer fails, create overnight payable as fallback
            payable_id = system.new_contract_id("P")
            overnight_payable = Payable(
                id=payable_id,
                kind="payable",
                amount=amount,
                denom="X",
                asset_holder_id=creditor_bank,
                liability_issuer_id=debtor_bank,
                due_day=day + 1
            )

            system.add_contract(overnight_payable)
            system.log("InterbankOvernightCreated",
                      debtor_bank=debtor_bank,
                      creditor_bank=creditor_bank,
                      amount=amount,
                      payable_id=payable_id,
                      due_day=day + 1)
