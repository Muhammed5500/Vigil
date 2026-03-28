import json
import os
from pathlib import Path
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()


class ChainService:
    def __init__(self):
        rpc_url = os.getenv("MONAD_RPC_URL", "https://testnet-rpc.monad.xyz")
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))

        # Load contract ABIs from hardhat artifacts
        artifacts_dir = Path(__file__).parent.parent / "contracts" / "artifacts" / "contracts"

        self.registry_abi = self._load_abi(artifacts_dir / "NodeRegistry.sol" / "NodeRegistry.json")
        self.vault_abi = self._load_abi(artifacts_dir / "StakeVault.sol" / "StakeVault.json")
        self.engine_abi = self._load_abi(artifacts_dir / "ScoringEngine.sol" / "ScoringEngine.json")

        # Contract addresses from env
        registry_addr = os.getenv("NODE_REGISTRY_ADDRESS", "")
        vault_addr = os.getenv("STAKE_VAULT_ADDRESS", "")
        engine_addr = os.getenv("SCORING_ENGINE_ADDRESS", "")

        if registry_addr:
            self.registry = self.w3.eth.contract(
                address=Web3.to_checksum_address(registry_addr), abi=self.registry_abi
            )
        if vault_addr:
            self.vault = self.w3.eth.contract(
                address=Web3.to_checksum_address(vault_addr), abi=self.vault_abi
            )
        if engine_addr:
            self.engine = self.w3.eth.contract(
                address=Web3.to_checksum_address(engine_addr), abi=self.engine_abi
            )

        # Coordinator wallet
        self.coordinator_key = os.getenv("PRIVATE_KEY_COORDINATOR", "")
        if self.coordinator_key:
            self.coordinator_account = self.w3.eth.account.from_key(self.coordinator_key)
            self.coordinator_address = self.coordinator_account.address

    def _load_abi(self, path):
        try:
            with open(path) as f:
                return json.load(f)["abi"]
        except FileNotFoundError:
            return []

    def _send_tx(self, func, private_key, value=0):
        account = self.w3.eth.account.from_key(private_key)
        tx = func.build_transaction({
            "from": account.address,
            "nonce": self.w3.eth.get_transaction_count(account.address),
            "gas": 500000,
            "gasPrice": self.w3.eth.gas_price,
            "value": value,
            "chainId": 10143,
        })
        signed = self.w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.transactionHash.hex()

    def register_node(self, private_key: str) -> str:
        minimum_stake = self.registry.functions.minimumStake().call()
        return self._send_tx(
            self.registry.functions.register(),
            private_key,
            value=minimum_stake,
        )

    def stake_node(self, private_key: str, amount: int) -> str:
        return self._send_tx(
            self.vault.functions.stake(), private_key, value=amount
        )

    def submit_scores(self, addresses: list, scores: list) -> str:
        checksum_addrs = [Web3.to_checksum_address(a) for a in addresses]
        return self._send_tx(
            self.engine.functions.submitRoundScores(checksum_addrs, scores),
            self.coordinator_key,
        )

    def get_stake(self, address: str) -> int:
        return self.vault.functions.getStake(
            Web3.to_checksum_address(address)
        ).call()

    def get_round(self) -> int:
        return self.engine.functions.getCurrentRound().call()


if __name__ == "__main__":
    cs = ChainService()
    print("Connected:", cs.w3.is_connected())
    print("ChainService OK")
