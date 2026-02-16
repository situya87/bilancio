import uuid


def new_id(prefix: str = "x") -> str:
    # short, sortable-ish id; fine for MVP
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


AgentId = str
InstrId = str
OpId = str
