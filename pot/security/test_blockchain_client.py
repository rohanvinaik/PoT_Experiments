from pot.prototypes.training_provenance_auditor import BlockchainClient
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider


def _client():
    provider = EthereumTesterProvider()
    w3 = Web3(provider)
    return BlockchainClient(web3=w3)


def test_round_trip():
    client = _client()
    tx_id = client.store_hash("deadbeef", {"epoch": 1})
    retrieved = client.retrieve_hash(tx_id)
    assert retrieved is not None
    assert retrieved["hash"] == "deadbeef"
    assert retrieved["metadata"] == {"epoch": 1}
    assert client.verify_hash("deadbeef", tx_id)
    assert not client.verify_hash("badbeef", tx_id)


def test_missing_transaction():
    client = _client()
    missing = client.retrieve_hash("0x" + "0" * 64)
    assert missing is None
    assert not client.verify_hash("deadbeef", "0x" + "0" * 64)
