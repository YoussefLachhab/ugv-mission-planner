import uuid
def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]
