# Decentralized AI Training Research

> Research Date: 2026-02-06 (Updated: 2026-02-08)
> Status: Complete

## Overview

Exploring blockchain-based solutions for distributed AI training. The core idea: use cryptocurrency tokens to incentivize GPU providers, creating a marketplace for computational power.

---

## Existing Projects

### Bittensor (TAO) ⭐

The most mature decentralized AI project.

**Architecture:**
- **Subnets:** Specialized networks for different AI tasks (inference, training, etc.)
- **Miners:** Provide compute resources
- **Validators:** Rank and reward miners
- **TAO Token:** Economic backbone

**Key Features:**
- Peer-to-peer intelligence marketplace
- dTAO (Dynamic TAO) launched Feb 2025 — governance upgrade
- Yuma Consensus for fair reward distribution
- 41% of block rewards to AI training incentives

**Use Cases:**
- Decentralized inference
- Model training
- Fine-tuning
- Agent infrastructure

**Website:** https://bittensor.com

### Render Network (RNDR)
- **What:** Decentralized GPU computing on Ethereum
- **How:** Marketplace connecting GPU providers with users needing compute
- **Use cases:** Rendering, AI computation
- **Website:** https://render.x.io

### io.net
- **What:** Decentralized physical infrastructure network for GPU sourcing
- **How:** Sources GPU power for AI/ML workloads
- **Website:** https://io.net

### Flock.io
- **What:** Federated learning + blockchain
- **How:** Privacy-preserving collaborative model development
- **Features:** Token-incentivized, modular network

### Deepnode (DN)
- **What:** Connects GPU resource providers with AI developers
- **How:** Democratizes access to GPU computing power

### ChainGPT (CGPT)
- **What:** AIVM framework integrating AI into blockchain
- **Features:** Decentralized model execution, training, AI agents infrastructure

---

## Cost Comparison

### Centralized Cloud (AWS/GCP)

| Service | GPU | Cost/hr | 8hr fine-tune |
|---------|-----|---------|---------------|
| AWS p4d | A100 40GB | ~$3.50 | $28 |
| GCP a2-highgpu | A100 80GB | ~$4.00 | $32 |
| Lambda Labs | A100 80GB | ~$1.89 | $15 |
| RunPod | A10G | ~$0.50 | $4 |

### Decentralized

| Platform | Comparable GPU | Est. Cost/hr | Notes |
|----------|----------------|--------------|-------|
| io.net | A100 equiv | ~$0.50-1.50 | Variable |
| Bittensor | Subnet-dependent | TAO tokens | Complex pricing |
| Render | Variable | RNDR tokens | More for rendering |

**Verdict:** Decentralized can be cheaper but less predictable.

---

## Key Questions for Molting

### 1. Can we use existing infrastructure?
- ✅ Bittensor subnets for training compute
- ✅ io.net for GPU access
- ⚠️ Reliability concerns for long training

### 2. Should we create something new?
- Specialized for agent personality training
- Community-owned infrastructure
- **Assessment:** Probably not — existing infra is sufficient

### 3. Technical Feasibility
- Fine-tuning workloads: ✅ Possible on Bittensor
- Latency: ⚠️ Higher than dedicated cloud
- Reliability: ⚠️ Variable node availability

### 4. Privacy Considerations
- Training data on decentralized nodes = privacy risk
- Personality data is sensitive
- **Recommendation:** Use decentralized for inference, not fine-tuning

---

## Relevance to Molting Project

### Realistic Assessment

For Phase 2-3 of Molting:

| Task | Recommended Platform | Reason |
|------|---------------------|--------|
| Fine-tuning | **Local or Cloud** | Privacy, reliability |
| Inference | Decentralized possible | Cost savings |
| Community models | Bittensor | Incentive alignment |

### Future Vision

Once Molting has a working local model:
1. Could contribute to Bittensor subnet
2. Earn TAO for providing inference
3. Use earnings for more compute
4. Create a self-sustaining loop

---

## Hypotheses

**H016:** Bittensor subnet for personality fine-tuning is viable but not practical due to privacy concerns

**H017:** Decentralized inference could reduce long-term operating costs by 50%+

---

## Next Steps

- [x] Research existing platforms
- [x] Cost comparison analysis
- [ ] Test io.net for basic inference
- [ ] Explore Bittensor validator/miner economics
- [ ] Community validation on Moltbook

---

## References

1. [Bittensor Whitepaper](https://bittensor.com/whitepaper)
2. [Bittensor Docs](https://docs.learnbittensor.org/)
3. [io.net](https://io.net)
4. [Bittensor Protocol Analysis (arXiv)](https://arxiv.org/html/2507.02951v1)
5. [dTAO Introduction](https://www.prestolabs.io/research/from-bitcoin-to-bittensor-the-next-monetary-primitive)

---

*Analysis by Cláudio for Project Molting*
