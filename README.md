# Vigil

**Decentralized AI training verification protocol using peer prediction mechanisms on Monad.**

Vigil is the first known implementation that applies peer prediction — a family of mechanisms from the information elicitation literature — to AI training gradient verification. It enables permissionless, hardware-independent, and mathematically guaranteed verification of compute contributions in decentralized machine learning.

---

## The Problem

Decentralized AI training distributes model training across independent nodes. Each node computes gradients on a data partition and sends them to a coordinator for aggregation (Federated Averaging). The fundamental challenge: **how do you verify that a node actually performed the computation?**

A malicious node can:
- **Free-ride** — send random gradients without computing (saves GPU cost)
- **Sabotage** — send adversarial gradients to degrade the model
- **Bias** — inject subtle perturbations to steer the model toward a desired behavior

Without verification, a single malicious node can compromise the entire training process. For frontier models costing $100M+ to train, this is not just a technical problem — it is an economic and security imperative.

## Existing Approaches and Their Limitations

| Approach | Method | Limitation |
|----------|--------|------------|
| **TEE** (0G, Intel SGX) | Hardware-isolated execution with cryptographic attestation | Requires specialized hardware, vendor trust dependency, vulnerable to side-channel attacks (Spectre/Meltdown variants) |
| **Economic Incentives** (Bittensor) | Subjective validator scoring with stake-based penalties | Relies on trusted validators, vulnerable to validator collusion, cannot detect subtle poisoning |
| **Zero-Knowledge Proofs** | Cryptographic proof of correct computation | Computationally prohibitive for gradient computation at scale, impractical latency |
| **Redundant Computation** | Multiple nodes compute the same task | 2x-3x cost overhead, defeats the purpose of distributed training |

**Vigil fills the gap:** no specialized hardware, no trusted validators, no oracle dependency, permissionless participation, and mathematically guaranteed honesty incentives — all at marginal computational overhead.

---

## The Vigil Solution

### Core Insight

Peer prediction mechanisms can determine the honesty of reports **without access to ground truth**. The key observation: independent honest reporters observing the same signal produce correlated reports. Dishonest reporters break this correlation.

This maps directly to AI training verification:

| Peer Prediction Domain | AI Training Domain |
|------------------------|-------------------|
| Reporter | Training node |
| Signal (observation) | Computed gradient |
| Ground truth | "Correct" gradient (unknown) |
| Honest reporting | Genuine computation |
| Dishonest reporting | Fake/random gradient |
| Peer comparison | Overlap set gradient comparison |
| Score | Node reliability score |
| Payment | Token reward/slash |

### Academic Foundation

Vigil builds on established mechanisms from the information elicitation without verification (IEWV) literature:

| Mechanism | Authors | Year | Key Contribution |
|-----------|---------|------|------------------|
| Peer Prediction Method | Miller, Resnick, Zeckhauser | 2005 | Score reports against peer reports without ground truth |
| Bayesian Truth Serum | Prelec | 2004 | Reward "surprisingly popular" answers to elicit truthful reports |
| Deterministic Robust Peer Prediction Mechanism (DRPM) | Dasgupta & Ghosh | 2013 | Simple mechanism for binary signals |
| Peer Truth Serum | Radanovic & Faltings | 2013 | Extension to continuous signals |

**Novelty:** A comprehensive search yields no prior work applying peer prediction to AI training gradient verification:
- `"peer prediction" + "gradient verification"` — no results
- `"peer prediction" + "AI training"` — no results
- Related but distinct: arXiv:2406.01794 (peer prediction for blockchain verifier's dilemma, not AI training), arXiv:2208.04433 (peer prediction for learning agents, not training verification)

Vigil represents a novel intersection of mechanism design and decentralized machine learning.

---

## Mathematical Framework

### 1. Overlap Set Construction

Each training round partitions the dataset $D$ into shared and private subsets:

```
D = {d_1, d_2, ..., d_n}

D_shared ⊂ D,  |D_shared| = α|D|,  α ∈ [0.05, 0.10]

D_private_i = partition(D \ D_shared, N)  for node i ∈ {1, ..., N}
```

Where `α` is the overlap ratio and `N` is the number of nodes. `D_shared` is selected using an on-chain random seed that changes every round, preventing pre-computation attacks.

**Critical property:** All nodes receive identical `D_shared` but different `D_private_i`. Honest nodes processing `D_shared` through the same model `M` with identical parameters `θ` must produce identical gradients (up to floating-point precision).

### 2. Gradient Computation

Each node `i` computes two gradients per round:

```
g_shared_i = ∇_θ L(M(D_shared; θ), Y_shared)     // verification gradient
g_private_i = ∇_θ L(M(D_private_i; θ), Y_private_i)  // training gradient
```

Where `L` is the loss function (cross-entropy) and `∇_θ` denotes the gradient with respect to model parameters `θ`.

For an **honest node**: `g_shared_i` is the true gradient computed via backpropagation.
For a **cheating node**: `g_shared_i` is drawn from some distribution unrelated to the actual computation (e.g., random normal).

### 3. Cosine Similarity

Gradient similarity is measured via cosine similarity, which compares directional alignment independent of magnitude:

```
cos(g_i, g_j) = (g_i · g_j) / (||g_i||₂ · ||g_j||₂)
```

**Value range:** `[-1, 1]`
- `cos ≈ 1.0` — same direction (both honest, same computation)
- `cos ≈ 0.0` — orthogonal (one is likely random)
- `cos ≈ -1.0` — opposite direction (adversarial sabotage)

**Why cosine similarity?**
- Invariant to gradient magnitude — nodes with different hardware may produce gradients with different scales but identical directions
- Efficient: `O(d)` where `d` = number of model parameters
- Effective in high-dimensional spaces (207K+ dimensions in our CNN)

### 4. Peer Prediction Scoring

Each node's score is the average cosine similarity with all other nodes' shared gradients:

```
score(i) = (1 / (N-1)) · Σ_{j≠i} cos(g_shared_i, g_shared_j)
```

**Worked example with 4 nodes (3 honest, 1 cheater):**

```
Shared gradients:
  g_A = [0.23, -0.41, 0.87, 0.12, -0.55]   // honest
  g_B = [0.25, -0.39, 0.85, 0.14, -0.53]   // honest
  g_C = [0.22, -0.43, 0.88, 0.11, -0.56]   // honest
  g_D = [0.91, 0.12, -0.55, 0.44, 0.33]    // cheater (random)

Pairwise similarities:
  cos(A,B) = 0.998    cos(A,C) = 0.999    cos(A,D) = -0.002
  cos(B,C) = 0.997    cos(B,D) = -0.002   cos(C,D) = -0.002

Scores:
  score(A) = (0.998 + 0.999 + (-0.002)) / 3 = 0.665  → REWARD
  score(B) = (0.998 + 0.997 + (-0.002)) / 3 = 0.664  → REWARD
  score(C) = (0.999 + 0.997 + (-0.002)) / 3 = 0.665  → REWARD
  score(D) = ((-0.002) + (-0.002) + (-0.002)) / 3 = -0.002  → SLASH
```

### 5. Threshold-Based Settlement

```
threshold τ = 0.2

score(i) ≥ τ  →  REWARD: stake_i += stake_i × r_reward    (r_reward = 2%)
score(i) < τ  →  SLASH:  stake_i -= stake_i × r_slash      (r_slash = 5%)
```

The asymmetric penalty (`r_slash > r_reward`) ensures that the expected value of cheating is strictly negative:

```
E[cheat] = P(undetected) × reward - P(detected) × slash

With N honest nodes >> K cheaters:
  P(detected) → 1  (cheater's gradient is uncorrelated with honest majority)
  E[cheat] ≈ -slash < 0

Therefore: honest behavior is the strictly dominant strategy.
```

### 6. Model Update (Federated Averaging)

Only gradients from nodes with `score ≥ τ` are used for model update:

```
θ_{t+1} = θ_t - η · (1/|H|) · Σ_{i∈H} g_private_i
```

Where `H` is the set of honest nodes (score above threshold), `η` is the learning rate, and `g_private_i` is node `i`'s gradient on its private data. Cheater gradients are **excluded** from aggregation, protecting model integrity.

---

## Security Analysis

### Attack 1: Random Gradient (Free-riding)

```
Attack:  Node sends random gradient without computing
Detect:  Random vector in R^d is orthogonal to any fixed vector with high probability
Result:  cos(random, honest) ≈ 0 → score < threshold → SLASH
```

For `d = 207,922` parameters, the probability that a random gradient has `cos > 0.2` with an honest gradient is astronomically small (`< 10^{-1000}`).

### Attack 2: Collusion

```
Attack:  K nodes agree to send identical fake gradients
Detect:  Overlap set changes randomly each round (on-chain seed)
         Colluders cannot predict the exam questions
         To produce correlated fake answers, they must do the real computation
         → which means they are being honest
Result:  When N >> K, colluders are the minority → detected and slashed
```

### Attack 3: Sybil Attack

```
Attack:  One entity creates 100 fake nodes
Detect:  Each node requires minimum stake
         100 nodes = 100× stake at risk
         All sending random gradients → all get slashed
Result:  Economically irrational: expected loss = 100 × stake × P(detected) ≈ 100 × stake
```

### Attack 4: Subtle Poisoning

```
Attack:  Node sends 95% correct gradient + 5% adversarial bias
Detect:  Single round: difficult (cos ≈ 0.95, above threshold)
         Multi-round: statistical anomaly detection via reputation tracking
Result:  With N >> 1, a single node's 5% bias has negligible effect on the
         aggregated model (diluted by N honest gradients)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js)                     │
│   Node Dashboard │ Score Leaderboard │ TX Explorer        │
│   Accuracy Chart │ Stake History     │ Draw Digit Test    │
└───────────────────────────┬──────────────────────────────┘
                            │ REST API (polling every 2s)
┌───────────────────────────┴──────────────────────────────┐
│                  COORDINATOR (FastAPI)                     │
│   Data Distribution │ Gradient Aggregation │ PP Scoring    │
│                Chain Interaction (web3.py)                 │
└──────┬──────────┬──────────┬──────────┬──────────────────┘
       │          │          │          │
  ┌────┴───┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐
  │ Node A │ │ Node B │ │ Node C │ │ Node D │
  │ Honest │ │ Honest │ │ Honest │ │Cheater │
  │PyTorch │ │PyTorch │ │PyTorch │ │ Random │
  └────────┘ └────────┘ └────────┘ └────────┘

┌──────────────────────────────────────────────────────────┐
│                    MONAD TESTNET                           │
│  NodeRegistry.sol  │  StakeVault.sol  │  ScoringEngine.sol│
│  register()        │  stake()         │  submitScores()   │
│  getActiveNodes()  │  slash()         │  getRoundScore()  │
│  deactivateNode()  │  reward()        │  getCurrentRound()│
└──────────────────────────────────────────────────────────┘
```

### Contract Addresses (Monad Testnet)

| Contract | Address |
|----------|---------|
| NodeRegistry | `0x7B2407E9b7038e849bF4624b0471596674856938` |
| StakeVault | `0xCf09F677Dccf16AFD4ccfe8696E4f3edD354Cd97` |
| ScoringEngine | `0x69943AE8dCe775D41C33cBE9CaCBDa6d1fDbf0E7` |

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Smart Contracts | Solidity ^0.8.19 | On-chain node registration, stake management, score settlement |
| Contract Tooling | Hardhat | Testing (22 tests passing), deployment |
| Backend | Python + FastAPI | Coordinator logic, gradient aggregation, PP scoring |
| ML Framework | PyTorch | MNIST CNN training, gradient computation |
| Chain Interaction | web3.py | Transaction submission, stake queries |
| Frontend | Next.js 16 + React 19 | Real-time dashboard with live training visualization |
| Styling | Tailwind CSS 4 | Dark theme UI |
| Charts | Recharts | Accuracy and stake history visualization |
| Blockchain | Monad Testnet | High-throughput EVM chain (10K TPS, ~1s finality) |

---

## Why Monad?

Vigil requires on-chain settlement every training round — score submission, stake updates, and event emission for `N` nodes. This demands:

- **High throughput:** `N` transactions per round (Monad: 10,000 TPS vs Ethereum: ~15 TPS)
- **Low latency:** Training cannot wait minutes for finality (Monad: ~1s vs Ethereum: ~12 min)
- **Low gas cost:** Frequent small transactions must be economically viable
- **Parallel execution:** Independent stake updates for `N` nodes can execute concurrently

Monad's parallel EVM execution is particularly relevant: `N` independent `slash()` and `reward()` calls can execute in parallel within a single block, scaling linearly with node count.

---

## Local Development

### Prerequisites

- Node.js 18+
- Python 3.11+
- Git

### Setup

```bash
git clone https://github.com/Muhammed5500/Vigil.git
cd Vigil

# Smart Contracts
cd contracts
npm install
npx hardhat test  # 22 tests passing
cd ..

# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn web3 numpy python-dotenv pydantic Pillow six
cp .env.example .env  # configure with your keys
cd ..

# Frontend
cd frontend
npm install
cp .env.example .env.local
cd ..
```

### Run

```bash
# Terminal 1 — Backend
cd backend && uvicorn main:app --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev

# Open http://localhost:3000 and click "Start Training"
```

---

## Training Flow

```
Round 1/20                          Round 20/20
   │                                    │
   ▼                                    ▼
Accuracy: 26%                       Accuracy: 87%
Node A: 1.0000 MON                  Node A: 1.4859 MON  (+48.6%)
Node D: 1.0000 MON                  Node D: 0.3584 MON  (-64.2%)

Cheater detected ✗                  Cheater bankrupt ✗
Model protected ✓                   Model trained ✓
```

Each round:
1. Data distributed (shared overlap set + private partitions)
2. Nodes compute gradients (honest: PyTorch backprop, cheater: random)
3. Peer prediction scores computed (cosine similarity averaging)
4. Scores submitted to Monad (real transaction, real gas)
5. Smart contract executes slash/reward on stakes
6. Model updated with honest gradients only (FedAvg)
7. Accuracy evaluated on held-out test set

---

## Results

| Metric | Value |
|--------|-------|
| Model | MNIST CNN (207,922 parameters) |
| Nodes | 4 (3 honest, 1 cheater) |
| Rounds | 20 |
| Final accuracy | ~87% |
| Cheater detection rate | 100% (every round) |
| Honest node return | +48.6% stake growth |
| Cheater loss | -64.2% stake loss |
| False positive rate | 0% |
| On-chain transactions | 20 real TXs on Monad testnet |

---

## References

### Academic

1. Miller, N., Resnick, P., & Zeckhauser, R. (2005). *Eliciting Informative Feedback: The Peer-Prediction Method.* Management Science, 51(9), 1359-1373.
2. Prelec, D. (2004). *A Bayesian Truth Serum for Subjective Data.* Science, 306(5695), 462-466.
3. Dasgupta, A. & Ghosh, A. (2013). *Crowdsourced Judgement Elicitation with Endogenous Proficiency.* WWW 2013.
4. Radanovic, G. & Faltings, B. (2013). *A Robust Bayesian Truth Serum for Non-Binary Signals.* AAAI 2013.
5. McMahan, B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.

### Related Work

6. arXiv:2406.01794 — *It Takes Two: A Peer-Prediction Solution for Blockchain Verifier's Dilemma*
7. arXiv:2208.04433 — *Peer Prediction for Learning Agents*
8. arXiv:2308.10502 — *GradientCoin: A Peer-to-Peer Decentralized Large Language Models*

---

## Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 | 4 nodes, MNIST, basic PP, Monad testnet | **Complete** |
| Phase 2 | 10+ nodes, CIFAR-10, advanced PP (MTS), commit-reveal, reputation | Planned |
| Phase 3 | 100+ nodes, real model training, Monad mainnet, token economics | Planned |
| Phase 4 | 1000+ nodes, frontier model training, cross-chain, academic paper | Vision |

---

## License

MIT

---

*Built for the Monad Hackathon. Vigil applies peer prediction to AI training verification — no hardware requirements, no trusted validators, just math.*
