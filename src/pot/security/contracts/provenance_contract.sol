// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ProofOfTrainingContract
 * @dev Simple provenance storage contract for PoT (Proof-of-Training) verification
 * 
 * This contract stores cryptographic hashes with metadata for neural network
 * training verification. It provides tamper-evident storage on blockchain
 * for the PoT framework's verification needs.
 */
contract ProofOfTrainingContract {
    // Contract owner
    address public owner;
    
    // Transaction counter for unique IDs
    uint256 private transactionCounter;
    
    // Hash record structure
    struct HashRecord {
        string hash;           // Cryptographic hash (hex string)
        string metadata;       // JSON metadata string
        uint256 timestamp;     // Block timestamp when stored
        uint256 blockNumber;   // Block number when stored
        address submitter;     // Address that submitted the hash
        bool exists;          // Flag to check if record exists
    }
    
    // Mapping from transaction ID to hash record
    mapping(uint256 => HashRecord) private hashRecords;
    
    // Events
    event HashStored(
        uint256 indexed transactionId,
        string indexed hashValue,
        address indexed submitter,
        uint256 timestamp,
        uint256 blockNumber
    );
    
    event ContractPaused(address indexed owner, uint256 timestamp);
    event ContractUnpaused(address indexed owner, uint256 timestamp);
    
    // Contract state
    bool public paused;
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    modifier validHash(string memory _hash) {
        require(bytes(_hash).length > 0, "Hash cannot be empty");
        require(bytes(_hash).length <= 128, "Hash too long");
        _;
    }
    
    modifier validMetadata(string memory _metadata) {
        require(bytes(_metadata).length <= 2048, "Metadata too long");
        _;
    }
    
    /**
     * @dev Constructor sets the contract owner
     */
    constructor() {
        owner = msg.sender;
        transactionCounter = 0;
        paused = false;
    }
    
    /**
     * @dev Store a cryptographic hash with metadata
     * @param _hash The cryptographic hash to store (hex string)
     * @param _metadata JSON metadata string associated with the hash
     * @return transactionId Unique identifier for this storage transaction
     */
    function storeHash(
        string memory _hash,
        string memory _metadata
    ) 
        public 
        whenNotPaused 
        validHash(_hash) 
        validMetadata(_metadata) 
        returns (uint256 transactionId) 
    {
        // Increment counter to generate unique transaction ID
        transactionCounter++;
        transactionId = transactionCounter;
        
        // Create hash record
        hashRecords[transactionId] = HashRecord({
            hash: _hash,
            metadata: _metadata,
            timestamp: block.timestamp,
            blockNumber: block.number,
            submitter: msg.sender,
            exists: true
        });
        
        // Emit event
        emit HashStored(
            transactionId,
            _hash,
            msg.sender,
            block.timestamp,
            block.number
        );
        
        return transactionId;
    }
    
    /**
     * @dev Retrieve a stored hash record by transaction ID
     * @param _transactionId The transaction ID to look up
     * @return hash The stored cryptographic hash
     * @return metadata The stored metadata JSON string
     * @return timestamp When the hash was stored
     * @return blockNumber Block number when hash was stored
     * @return submitter Address that submitted the hash
     */
    function getHash(uint256 _transactionId) 
        public 
        view 
        returns (
            string memory hash,
            string memory metadata,
            uint256 timestamp,
            uint256 blockNumber,
            address submitter
        ) 
    {
        require(hashRecords[_transactionId].exists, "Transaction ID does not exist");
        
        HashRecord memory record = hashRecords[_transactionId];
        return (
            record.hash,
            record.metadata,
            record.timestamp,
            record.blockNumber,
            record.submitter
        );
    }
    
    /**
     * @dev Verify that a hash matches the stored value for a transaction ID
     * @param _hash The hash to verify
     * @param _transactionId The transaction ID to check against
     * @return matches True if hash matches stored value, false otherwise
     */
    function verifyHash(string memory _hash, uint256 _transactionId) 
        public 
        view 
        returns (bool matches) 
    {
        if (!hashRecords[_transactionId].exists) {
            return false;
        }
        
        return keccak256(abi.encodePacked(_hash)) == 
               keccak256(abi.encodePacked(hashRecords[_transactionId].hash));
    }
    
    /**
     * @dev Get the total number of stored transactions
     * @return count The total transaction count
     */
    function getTransactionCount() public view returns (uint256 count) {
        return transactionCounter;
    }
    
    /**
     * @dev Check if a transaction ID exists
     * @param _transactionId The transaction ID to check
     * @return exists True if the transaction exists
     */
    function transactionExists(uint256 _transactionId) public view returns (bool exists) {
        return hashRecords[_transactionId].exists;
    }
    
    /**
     * @dev Get basic information about a transaction (without full data)
     * @param _transactionId The transaction ID to check
     * @return exists Whether the transaction exists
     * @return timestamp When it was stored
     * @return blockNumber Block number when stored
     * @return submitter Who submitted it
     */
    function getTransactionInfo(uint256 _transactionId) 
        public 
        view 
        returns (
            bool exists,
            uint256 timestamp,
            uint256 blockNumber,
            address submitter
        ) 
    {
        HashRecord memory record = hashRecords[_transactionId];
        return (
            record.exists,
            record.timestamp,
            record.blockNumber,
            record.submitter
        );
    }
    
    // Owner-only management functions
    
    /**
     * @dev Pause the contract (owner only)
     * Prevents new hash storage but allows reading existing data
     */
    function pauseContract() public onlyOwner {
        require(!paused, "Contract is already paused");
        paused = true;
        emit ContractPaused(msg.sender, block.timestamp);
    }
    
    /**
     * @dev Unpause the contract (owner only)
     */
    function unpauseContract() public onlyOwner {
        require(paused, "Contract is not paused");
        paused = false;
        emit ContractUnpaused(msg.sender, block.timestamp);
    }
    
    /**
     * @dev Transfer ownership to a new address (owner only)
     * @param _newOwner The address of the new owner
     */
    function transferOwnership(address _newOwner) public onlyOwner {
        require(_newOwner != address(0), "New owner cannot be zero address");
        require(_newOwner != owner, "New owner must be different from current owner");
        owner = _newOwner;
    }
    
    /**
     * @dev Emergency function to withdraw any accidentally sent ETH (owner only)
     * Note: This contract should not normally receive ETH
     */
    function emergencyWithdraw() public onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH to withdraw");
        payable(owner).transfer(balance);
    }
    
    /**
     * @dev Get contract metadata and statistics
     * @return contractOwner The owner address
     * @return totalTransactions Total number of transactions stored
     * @return contractPaused Whether the contract is paused
     * @return contractBalance ETH balance (should normally be 0)
     */
    function getContractInfo() 
        public 
        view 
        returns (
            address contractOwner,
            uint256 totalTransactions,
            bool contractPaused,
            uint256 contractBalance
        ) 
    {
        return (
            owner,
            transactionCounter,
            paused,
            address(this).balance
        );
    }
    
    // Fallback function to reject direct ETH transfers
    receive() external payable {
        revert("This contract does not accept ETH");
    }
    
    fallback() external payable {
        revert("Function not found");
    }
}