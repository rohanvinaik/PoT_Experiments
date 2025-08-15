import hashlib, secrets

def commit_message(challenge_id: str, salt_hex: str) -> str:
    return hashlib.sha256((challenge_id + salt_hex).encode()).hexdigest()

def new_session_nonce() -> str:
    return secrets.token_hex(16)