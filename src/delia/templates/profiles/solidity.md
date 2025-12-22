# Solidity/Web3 Profile

Load this profile for: Smart contracts, Ethereum, DeFi, blockchain development.

## Contract Structure

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/// @title Token Vault
/// @author Your Name
/// @notice Secure token storage with access control
/// @dev Implements CEI pattern and reentrancy protection
contract TokenVault is Ownable, ReentrancyGuard, Pausable {
    // Type declarations
    struct Deposit {
        uint256 amount;
        uint256 timestamp;
    }

    // State variables
    mapping(address => Deposit) private deposits;
    uint256 public totalDeposits;

    // Events
    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    // Errors (gas efficient)
    error InsufficientBalance(uint256 available, uint256 required);
    error InvalidAmount();

    // Modifiers
    modifier validAmount(uint256 amount) {
        if (amount == 0) revert InvalidAmount();
        _;
    }
}
```

## Security Patterns

```solidity
// Checks-Effects-Interactions (CEI)
function withdraw(uint256 amount) external nonReentrant {
    // CHECKS
    Deposit storage deposit = deposits[msg.sender];
    if (deposit.amount < amount) {
        revert InsufficientBalance(deposit.amount, amount);
    }

    // EFFECTS (state changes before external calls)
    deposit.amount -= amount;
    totalDeposits -= amount;

    // INTERACTIONS (external calls last)
    (bool success, ) = msg.sender.call{value: amount}("");
    require(success, "Transfer failed");

    emit Withdrawn(msg.sender, amount);
}

// Pull over Push pattern
mapping(address => uint256) public pendingWithdrawals;

function claimRewards() external {
    uint256 amount = pendingWithdrawals[msg.sender];
    pendingWithdrawals[msg.sender] = 0;

    (bool success, ) = msg.sender.call{value: amount}("");
    require(success, "Transfer failed");
}
```

## Access Control

```solidity
import "@openzeppelin/contracts/access/AccessControl.sol";

contract MyContract is AccessControl {
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    function adminFunction() external onlyRole(ADMIN_ROLE) {
        // Admin-only logic
    }
}
```

## Gas Optimization

```solidity
// Use custom errors instead of revert strings
error Unauthorized();  // Cheaper than require(false, "Unauthorized")

// Pack storage variables
struct User {
    uint128 balance;     // Slot 0
    uint64 lastUpdate;   // Slot 0
    uint64 nonce;        // Slot 0
    address wallet;      // Slot 1
}

// Use immutable for constructor-set values
address public immutable factory;
uint256 public immutable fee;

// Use unchecked for safe math
function increment(uint256 x) internal pure returns (uint256) {
    unchecked { return x + 1; }  // Safe if overflow impossible
}

// Cache storage in memory
function processUsers(address[] calldata users) external {
    uint256 length = users.length;  // Cache length
    for (uint256 i = 0; i < length;) {
        // Process user
        unchecked { ++i; }  // Cheaper than i++
    }
}
```

## Upgradeable Contracts

```solidity
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";

contract MyContractV1 is Initializable {
    uint256 public value;

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(uint256 _value) public initializer {
        value = _value;
    }
}
```

## Testing (Foundry)

```solidity
// test/TokenVault.t.sol
import "forge-std/Test.sol";

contract TokenVaultTest is Test {
    TokenVault vault;
    address user = address(0x1);

    function setUp() public {
        vault = new TokenVault();
        vm.deal(user, 10 ether);
    }

    function testDeposit() public {
        vm.prank(user);
        vault.deposit{value: 1 ether}();
        assertEq(vault.balanceOf(user), 1 ether);
    }

    function testFuzz_Deposit(uint256 amount) public {
        vm.assume(amount > 0 && amount <= 10 ether);
        vm.prank(user);
        vault.deposit{value: amount}();
        assertEq(vault.balanceOf(user), amount);
    }
}
```

## Security Checklist

```
BEFORE DEPLOYMENT:
□ Reentrancy protection on external calls
□ Access control on sensitive functions
□ Input validation on all parameters
□ Integer overflow checks (use 0.8+)
□ Static analysis (Slither, Mythril)
□ Unit + fuzz tests with high coverage
□ Professional audit for mainnet
```

## Best Practices

```
ALWAYS:
- Use Checks-Effects-Interactions (CEI) pattern
- Use ReentrancyGuard for external calls
- Use custom errors for gas efficiency
- Use immutable for constructor-set values
- Emit events for all state changes
- Get professional audit before mainnet

AVOID:
- External calls before state changes
- Using tx.origin for authorization
- Unchecked arithmetic in Solidity <0.8
- Hardcoded addresses
- Storing secrets on-chain
- Complex logic in fallback functions
```

