# ProofOfTraining Smart Contract

This directory contains the smart contract implementation for storing cryptographic hashes on-chain for the PoT (Proof-of-Training) verification framework.

## Contract Overview

The `ProofOfTrainingContract` is a simple Solidity contract that provides tamper-evident storage for cryptographic hashes and their associated metadata on Ethereum-compatible blockchains.

### Key Features

- **Hash Storage**: Store cryptographic hashes with JSON metadata
- **Verification**: Verify stored hashes against provided values
- **Event Logging**: Emit events for all storage operations
- **Owner Management**: Pause/unpause functionality and ownership transfer
- **Gas Optimized**: Minimal storage and computation overhead

### Core Functions

- `storeHash(string hash, string metadata) → uint256`: Store a hash with metadata
- `getHash(uint256 transactionId) → (string, string, uint256, uint256, address)`: Retrieve stored data
- `verifyHash(string hash, uint256 transactionId) → bool`: Verify hash matches stored value
- `getTransactionCount() → uint256`: Get total number of stored transactions

## Files

- `provenance_contract.sol`: Main Solidity contract source code
- `abi.json`: Compiled Application Binary Interface for Web3 integration
- `README.md`: This documentation file

## Compilation

### Prerequisites

- [Solidity Compiler](https://docs.soliditylang.org/en/latest/installing-solidity.html) (solc) version 0.8.19+
- [Node.js](https://nodejs.org/) and npm
- [Hardhat](https://hardhat.org/) or [Truffle](https://trufflesuite.com/) (recommended)

### Option 1: Using Hardhat (Recommended)

1. **Initialize Hardhat project** (if not already done):
   ```bash
   npm init -y
   npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox
   npx hardhat init
   ```

2. **Copy contract to Hardhat contracts directory**:
   ```bash
   cp provenance_contract.sol contracts/
   ```

3. **Compile the contract**:
   ```bash
   npx hardhat compile
   ```

4. **Extract ABI** (generated in `artifacts/contracts/ProofOfTrainingContract.sol/ProofOfTrainingContract.json`):
   ```bash
   # Update abi.json with the compiled ABI
   jq '.abi' artifacts/contracts/ProofOfTrainingContract.sol/ProofOfTrainingContract.json > abi.json
   ```

### Option 2: Using Solidity Compiler Directly

```bash
# Compile contract
solc --abi --bin --optimize --overwrite -o build/ provenance_contract.sol

# The ABI will be in build/ProofOfTrainingContract.abi
cp build/ProofOfTrainingContract.abi abi.json
```

### Option 3: Using Remix IDE

1. Open [Remix IDE](https://remix.ethereum.org/)
2. Create new file and paste the contract code
3. Compile using the Solidity compiler plugin
4. Copy the ABI from the compilation artifacts

## Deployment

### Environment Setup

Set up your environment variables:

```bash
# Network RPC URLs
export ETHEREUM_RPC_URL="https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
export POLYGON_RPC_URL="https://polygon-rpc.com"
export GOERLI_RPC_URL="https://goerli.infura.io/v3/YOUR_PROJECT_ID"
export MUMBAI_RPC_URL="https://rpc-mumbai.maticvigil.com"

# Deployer private key (keep secure!)
export DEPLOYER_PRIVATE_KEY="0x1234..."

# Etherscan API keys for verification
export ETHERSCAN_API_KEY="YOUR_ETHERSCAN_API_KEY"
export POLYGONSCAN_API_KEY="YOUR_POLYGONSCAN_API_KEY"
```

### Deployment Scripts

#### Hardhat Deployment

Create `scripts/deploy.js`:

```javascript
async function main() {
  const [deployer] = await ethers.getSigners();
  
  console.log("Deploying with account:", deployer.address);
  console.log("Account balance:", (await deployer.getBalance()).toString());
  
  const ProofOfTrainingContract = await ethers.getContractFactory("ProofOfTrainingContract");
  const contract = await ProofOfTrainingContract.deploy();
  
  await contract.deployed();
  
  console.log("ProofOfTrainingContract deployed to:", contract.address);
  console.log("Transaction hash:", contract.deployTransaction.hash);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

Deploy to different networks:

```bash
# Deploy to local network
npx hardhat run scripts/deploy.js --network localhost

# Deploy to Goerli testnet
npx hardhat run scripts/deploy.js --network goerli

# Deploy to Polygon mainnet
npx hardhat run scripts/deploy.js --network polygon

# Deploy to Mumbai testnet
npx hardhat run scripts/deploy.js --network mumbai
```

#### Direct Deployment with Web3.py

```python
from web3 import Web3
from eth_account import Account
import json

# Setup
w3 = Web3(Web3.HTTPProvider("YOUR_RPC_URL"))
account = Account.from_key("YOUR_PRIVATE_KEY")

# Load compiled contract
with open('abi.json', 'r') as f:
    abi = json.load(f)

with open('build/ProofOfTrainingContract.bin', 'r') as f:
    bytecode = f.read()

# Deploy
contract = w3.eth.contract(abi=abi, bytecode=bytecode)
transaction = contract.constructor().build_transaction({
    'from': account.address,
    'gas': 2000000,
    'gasPrice': w3.to_wei('30', 'gwei'),
    'nonce': w3.eth.get_transaction_count(account.address)
})

signed_txn = account.sign_transaction(transaction)
tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print(f"Contract deployed at: {receipt.contractAddress}")
```

## Network Deployment Addresses

Update this section after deployment:

### Mainnet
- **Ethereum**: `0x...` (Not deployed)
- **Polygon**: `0x...` (Not deployed)

### Testnet
- **Goerli**: `0x...` (Not deployed)
- **Mumbai**: `0x...` (Not deployed)

## Contract Verification

### Etherscan Verification

```bash
# Verify on Etherscan (Ethereum)
npx hardhat verify --network mainnet DEPLOYED_CONTRACT_ADDRESS

# Verify on Polygonscan (Polygon)
npx hardhat verify --network polygon DEPLOYED_CONTRACT_ADDRESS
```

### Manual Verification

1. Go to the appropriate block explorer:
   - Ethereum: [etherscan.io](https://etherscan.io)
   - Polygon: [polygonscan.com](https://polygonscan.com)
   - Goerli: [goerli.etherscan.io](https://goerli.etherscan.io)
   - Mumbai: [mumbai.polygonscan.com](https://mumbai.polygonscan.com)

2. Navigate to your contract address
3. Click "Contract" → "Verify and Publish"
4. Select compiler version: `0.8.19+`
5. Select optimization: `Yes` (if used during compilation)
6. Paste the contract source code
7. Submit for verification

## Integration with PoT Framework

After deployment, update your PoT configuration:

```bash
# Set environment variables for Web3BlockchainClient
export RPC_URL="https://polygon-rpc.com"
export CONTRACT_ADDRESS="0xYOUR_DEPLOYED_CONTRACT_ADDRESS"
export PRIVATE_KEY="0xYOUR_PRIVATE_KEY"
```

Update `web3_blockchain_client.py` if needed to use the deployed contract address.

## Gas Cost Estimates

Approximate gas costs on different networks:

| Operation | Gas Used | Ethereum ($) | Polygon ($) |
|-----------|----------|--------------|-------------|
| Deploy | ~1,500,000 | $30-150 | $0.01-0.05 |
| storeHash | ~80,000 | $1.5-8 | $0.001-0.01 |
| getHash | ~30,000 | $0.5-3 | Free (read) |
| verifyHash | ~25,000 | $0.4-2.5 | Free (read) |

*Costs vary with network congestion and gas prices*

## Security Considerations

1. **Private Key Security**: Never commit private keys to version control
2. **Contract Ownership**: The deployer becomes the contract owner
3. **Pause Functionality**: Only owner can pause/unpause the contract
4. **Hash Validation**: Contract validates hash length and format
5. **Metadata Limits**: Metadata is limited to 2048 bytes to prevent abuse
6. **No ETH Storage**: Contract rejects direct ETH transfers

## Testing

Before mainnet deployment, thoroughly test on testnets:

1. **Deploy to testnet** (Goerli, Mumbai)
2. **Test all functions** using the PoT framework
3. **Verify events** are emitted correctly
4. **Test owner functions** (pause/unpause, ownership transfer)
5. **Load testing** with multiple transactions
6. **Gas optimization** testing

## Troubleshooting

### Common Issues

1. **Compilation Errors**:
   - Check Solidity version (requires 0.8.19+)
   - Ensure proper import paths

2. **Deployment Failures**:
   - Check account balance for gas fees
   - Verify network RPC URL is correct
   - Ensure private key has proper format

3. **Verification Issues**:
   - Match exact compiler version and settings
   - Include all import files if using libraries

4. **Transaction Failures**:
   - Check gas limit and gas price
   - Verify contract is not paused
   - Ensure hash format is valid

### Support

For additional support:
- Check the PoT framework documentation
- Review Ethereum/Polygon network status
- Consult Hardhat/Truffle documentation for deployment issues

## License

MIT License - See the contract header for full license information.