# Confidence Score Explained

## 📊 How Confidence is Calculated

The API returns a **confidence score** (0 to 1) for each prediction, indicating how certain the model is about its prediction.

### Formula: Weighted Confidence

```python
confidence = 0.7 × max_probability + 0.3 × margin

where:
- max_probability = highest class probability
- margin = difference between top 2 probabilities
```

This combines:
- **70% weight** on the maximum probability (how strong the top prediction is)
- **30% weight** on the margin (how much it beats the second-best option)

## 🎯 Confidence Interpretation

| Confidence Range | Interpretation | Example Probabilities |
|------------------|----------------|-----------------------|
| **0.90 - 1.00** | Very certain | `[0.95, 0.03, 0.02]` → 94% confidence |
| **0.70 - 0.90** | Quite certain | `[0.80, 0.15, 0.05]` → 76% confidence |
| **0.60 - 0.70** | Moderately certain | `[0.70, 0.20, 0.10]` → 64% confidence |
| **0.50 - 0.60** | Somewhat uncertain | `[0.65, 0.35, 0.00]` → 55% confidence |
| **0.40 - 0.50** | Uncertain | `[0.55, 0.40, 0.05]` → 43% confidence |
| **0.20 - 0.40** | Very uncertain | `[0.45, 0.40, 0.15]` → 33% confidence |
| **0.00 - 0.20** | No clear answer | `[0.34, 0.33, 0.33]` → 24% confidence |

## 📈 Confidence vs Probabilities Examples

### High Confidence Example
```json
{
  "predicted_class": 2,
  "class_probabilities": {
    "class_0": 0.03,
    "class_1": 0.05,
    "class_2": 0.92
  },
  "confidence": 0.92
}
```
**Interpretation:** Model is very certain this is a homopolymer (class 2)

### Medium Confidence Example
```json
{
  "predicted_class": 0,
  "class_probabilities": {
    "class_0": 0.65,
    "class_1": 0.35,
    "class_2": 0.00
  },
  "confidence": 0.55
}
```
**Interpretation:** Model predicts alternating (class 0), but there's significant uncertainty between class 0 and 1

### Low Confidence Example
```json
{
  "predicted_class": 1,
  "class_probabilities": {
    "class_0": 0.40,
    "class_1": 0.45,
    "class_2": 0.15
  },
  "confidence": 0.33
}
```
**Interpretation:** Model is very uncertain; probabilities are spread across multiple classes

## 🔄 Change from Previous Version

**Previous (Entropy-based):**
- Very strict, often gave 35-40% confidence for 65/35 splits
- Problem: Most predictions had similar low confidence scores
- Formula: `1 - (entropy / max_entropy)`

**Current (Weighted):**
- More intuitive and meaningful differentiation
- Better reflects the actual certainty of the model
- 65/35 split now gives ~55% confidence (instead of 40%)

## 💡 How to Use Confidence

### Recommended Actions by Confidence Level

| Confidence | Recommended Action |
|------------|-------------------|
| **> 80%** | High trust - use prediction directly |
| **60-80%** | Good trust - use with minor caution |
| **50-60%** | Moderate trust - consider context |
| **40-50%** | Low trust - validate with additional data |
| **< 40%** | Very low trust - prediction is uncertain, consider alternative approaches |

### Example Decision Logic

```python
if confidence > 0.7:
    # High confidence - trust the prediction
    print(f"Predicted: {predicted_class} (high confidence)")
elif confidence > 0.5:
    # Medium confidence - use with caution
    print(f"Predicted: {predicted_class} (moderate confidence)")
    print("Consider the second-best class as well")
else:
    # Low confidence - be careful
    print(f"Prediction uncertain (confidence: {confidence:.2f})")
    print("Review all class probabilities before deciding")
```

## 🔬 Technical Details

The weighted confidence metric was chosen because:

1. **Interpretable:** Directly related to the probabilities
2. **Balanced:** Considers both the winning probability and the margin
3. **Differentiating:** Provides meaningful spread across predictions
4. **Practical:** Aligns with how experts think about certainty

**Why not just use max probability?**
- `[0.51, 0.49, 0.00]` would give 51% confidence (too high for a close call)

**Why not just use margin?**
- `[0.40, 0.30, 0.30]` would give 10% confidence (too low when leading class is clear)

**Weighted approach:**
- `[0.51, 0.49, 0.00]` → ~50% confidence ✓ (uncertain)
- `[0.40, 0.30, 0.30]` → ~31% confidence ✓ (very uncertain)
- `[0.90, 0.05, 0.05]` → ~94% confidence ✓ (very certain)

## 📚 References

For more information about the prediction model and classes, see:
- `README.md` - API documentation
- Model classes: alternating (<1), random to block like (1-25), homopolymer (>25)
