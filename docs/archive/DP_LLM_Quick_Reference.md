# Differential Privacy for LLMs - Quick Reference

## 🎯 Core Findings (TL;DR)

1. **DP at inference for text generation = BAD** ❌
   - Unreadable output even with weak privacy (ε=15)
   - High dimensionality + sequential generation = fundamental failure
   
2. **DP-SGD (training time) = GOOD** ✅
   - Add DP during training, normal inference
   - Perfect text quality, strong privacy guarantees
   
3. **DP embeddings = GOOD** ✅  
   - Works great for search/clustering
   - Lower dimensionality (768 vs 50k)

4. **Query clustering is CRITICAL** ⚠️
   - Can only aggregate similar queries
   - Diverse queries → gibberish output

---

## 📊 Quick Comparison Table

| Approach | Privacy | Quality | When to Use |
|----------|---------|---------|-------------|
| DP-SGD | ✅ Strong | ✅ Perfect | Fine-tuning LLMs |
| DP Embeddings | ✅ Good | ✅ Good | Search/clustering |
| DP at Inference | ❌ Weak | ❌ Poor | Don't use (educational only) |

---

## 🔑 Key Equations

**Gaussian Mechanism**:
```
σ = sensitivity × √(2 × ln(1.25/δ)) / ε
noisy_output = true_output + N(0, σ²)
```

**Privacy-Utility Trade-off**:
- ε < 1: Very private, unusable quality
- ε = 1-3: Private, poor quality  
- ε = 5-10: Moderate privacy, marginal quality
- ε > 15: Weak privacy, acceptable quality

---

## 💡 Practical Recommendations

**Scenario: Users with sensitive queries**

### Option 1: DP-SGD (BEST) ⭐
```python
from opacus import PrivacyEngine

# 1. Collect private queries
# 2. Fine-tune with DP-SGD
privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private(
    module=model, optimizer=optimizer, data_loader=loader,
    noise_multiplier=1.1, max_grad_norm=1.0
)

# 3. Train normally
# 4. Deploy - inference is normal!
```

### Option 2: DP Embeddings (for search)
```python
# 1. Get embeddings
embeddings = model.encode(queries)

# 2. Add DP noise
σ = 2 * np.sqrt(2 * np.log(1.25/1e-5)) / epsilon
private_emb = embeddings + np.random.normal(0, σ, embeddings.shape)

# 3. Use for search/clustering
```

### Option 3: DP at Inference (AVOID)
- Only works with ε > 15 (weak privacy)
- Requires similar queries (clustering challenge)
- Poor quality, not production-ready

---

## 🚫 Common Pitfalls

1. **Using DP at inference for generation** - Won't work well
2. **Aggregating diverse queries** - Produces gibberish
3. **Top-k too large (100+)** - Noise affects too many dimensions
4. **Epsilon too low (<10)** - Text is unreadable
5. **Not clustering first** - Mixed queries = bad output

---

## ✅ Success Criteria

**For DP-SGD**:
- Epsilon: 1-10
- Delta: 10⁻⁵
- Model accuracy drop: <5%
- Inference: Normal speed

**For DP Embeddings**:
- Epsilon: 0.5-2
- Delta: 10⁻⁵  
- Cosine similarity preserved: >0.8
- Clustering quality: >80% of original

---

## 📚 Essential Resources

- **Opacus**: https://opacus.ai/
- **OpenDP**: https://docs.opendp.org/
- **Paper**: "Deep Learning with Differential Privacy" (Abadi et al., 2016)
- **Tutorial**: Google DP Blog

---

## 🎓 What We Discovered

### The Journey:
1. Started with DP at inference (standard approach)
2. Got gibberish output (unicode/multilingual garbage)
3. Fixed sampling (smaller top-k, higher ε, lower temp)
4. Still poor quality - discovered fundamental limitations
5. Realized similar queries needed - clustering challenge
6. Learned DP-SGD is the real solution

### The Lessons:
- High-dimensional + sequential = DP doesn't work
- Privacy-utility trade-off is harsh for text
- Add privacy during training, not inference
- Embeddings work better than generation

---

## 🔧 Fixed Sampling Function (if you must try)

```python
def sample_token_FIXED(dp_logits, temperature=0.4, top_k=15, top_p=0.85):
    """Much better than original - but still limited."""
    logits = torch.tensor(dp_logits) / temperature
    
    # Very aggressive top-k (15, not 100!)
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, float('-inf'))
    filtered[top_k_indices] = top_k_logits
    
    # Top-p filtering
    sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
    remove = cumulative_probs > top_p
    remove[0] = False
    filtered[sorted_indices[remove]] = float('-inf')
    
    # Sample
    probs = torch.softmax(filtered, -1)
    return torch.multinomial(probs, 1).item()

# Use with ε ≥ 15 for any hope of readability
```

---

## 📈 Privacy Budget Guidelines

| Use Case | Recommended ε | Recommended δ | Notes |
|----------|---------------|---------------|-------|
| Medical records | 0.5-1.0 | 10⁻⁶ | Very sensitive |
| Financial data | 1.0-3.0 | 10⁻⁵ | Sensitive |
| User queries | 3.0-10.0 | 10⁻⁵ | Moderately sensitive |
| Public data | >10.0 | 10⁻⁵ | Less sensitive |

**Note**: For text generation at inference, need ε > 15 for readability (weak privacy!)

---

## 🎯 Decision Tree

```
Need to use LLM with private data?
│
├─ Fine-tuning on private data?
│  └─> Use DP-SGD ✅
│
├─ Search/clustering only?
│  └─> Use DP Embeddings ✅
│
├─ Text generation from aggregated queries?
│  ├─ Queries similar?
│  │  ├─ Can accept ε > 15?
│  │  │  └─> Try DP at inference ⚠️
│  │  └─> No → Use DP-SGD on training data ✅
│  └─ Queries diverse?
│     └─> Cluster first, then decide ⚠️
│
└─ Just using API (GPT-4, etc)?
   └─> Can't add DP yourself (provider's responsibility)
```

---

## 🏆 Bottom Line

**The Answer**: **Use DP-SGD for training, not DP at inference!**

- ✅ Privacy: Strong (ε = 1-10)
- ✅ Quality: Perfect (no noise at inference)
- ✅ Speed: Normal inference speed
- ✅ Production-ready: Used by major companies

**Don't fight the fundamental limits of DP text generation at inference time!**

