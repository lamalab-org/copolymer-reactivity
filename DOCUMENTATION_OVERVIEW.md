# Documentation Overview

Complete guide to all documentation in this project.

## 📚 Documentation Structure

### Central Documentation
- **[Main README](README.md)** - Project overview, quick start, and navigation

### Component-Specific Documentation

#### 1. Prediction API (Production)
- **[API README](copol_prediction/api/README.md)** - Complete API documentation
- **[API Setup Guide](copol_prediction/api/SETUP_COMPLETE.md)** - Setup summary
- **Location**: `copol_prediction/api/`
- **Purpose**: REST API for making predictions

#### 2. ML Prediction Pipeline
- **[Pipeline README](copol_prediction/README.md)** - ML training & evaluation
- **Location**: `copol_prediction/`
- **Purpose**: Model training, analysis, and evaluation

#### 3. Data Extraction
- **[Extraction README](data_extraction/README.md)** - Literature data extraction
- **Location**: `data_extraction/`
- **Purpose**: Extracting data from scientific papers

#### 4. Experiments
- **[Experiments README](experiments/README.md)** - Baseline comparisons
- **Location**: `experiments/`
- **Purpose**: Testing different model configurations

## 🎯 Quick Navigation

### For Users

**I want to make predictions:**
→ Read [`copol_prediction/api/README.md`](copol_prediction/api/README.md)
→ Start: `cd copol_prediction/api && docker compose up`

**I want to use the Python library:**
→ Read [Main README](README.md) - "Usage Examples" section
→ See `src/copolpredictor/` for library code

**I want to understand the model:**
→ Read [`copol_prediction/README.md`](copol_prediction/README.md)
→ Check `copol_prediction/output/analysis/` for plots

### For Developers

**I want to train a new model:**
→ Read [`copol_prediction/README.md`](copol_prediction/README.md)
→ Run: `cd copol_prediction && python train_final_model.py`

**I want to extract new data:**
→ Read [`data_extraction/README.md`](data_extraction/README.md)
→ Run: `cd data_extraction && python obtain_data.py`

**I want to run experiments:**
→ Read [`experiments/README.md`](experiments/README.md)
→ Run: `cd experiments && ./run_all.sh`

**I want to modify the API:**
→ Read [`copol_prediction/api/README.md`](copol_prediction/api/README.md)
→ Edit: `copol_prediction/api/app.py`

### For Researchers

**I want to understand the methodology:**
→ Read [Main README](README.md) - "Model Performance" section
→ Read [`copol_prediction/README.md`](copol_prediction/README.md) - "Model Pipeline"

**I want to reproduce results:**
→ Read [`experiments/README.md`](experiments/README.md)
→ Run: `cd experiments && python sweep_filters.py`

**I want to analyze the model:**
→ Read [`copol_prediction/README.md`](copol_prediction/README.md) - "Analysis" section
→ Run: `cd copol_prediction && python analysis/analyze_model.py --all`

## 📖 Documentation Files

### README Files
| File | Purpose |
|------|---------|
| `/README.md` | Main project overview |
| `/copol_prediction/README.md` | ML pipeline documentation |
| `/copol_prediction/api/README.md` | REST API documentation |
| `/copol_prediction/api/SETUP_COMPLETE.md` | API setup summary |
| `/data_extraction/README.md` | Data extraction guide |
| `/experiments/README.md` | Experiments documentation |

### Configuration Files
| File | Purpose |
|------|---------|
| `/pyproject.toml` | Package configuration |
| `/LICENSE` | Project license (MIT) |
| `/copol_prediction/api/requirements.txt` | API dependencies |

### Example Files
| File | Purpose |
|------|---------|
| `/copol_prediction/api/example_client.py` | Python API client examples |
| `/copol_prediction/api/test_api.py` | API tests |

## 🔍 Finding Information

### Common Questions

**Q: How do I install this project?**
A: See [Main README](README.md) - "Installation" section

**Q: How do I use the API?**
A: See [`copol_prediction/api/README.md`](copol_prediction/api/README.md)

**Q: How do I train a model?**
A: See [`copol_prediction/README.md`](copol_prediction/README.md) - "Training" section

**Q: What features does the model use?**
A: See [Main README](README.md) - "Required Features" section

**Q: How accurate is the model?**
A: See [Main README](README.md) - "Model Performance" section

**Q: How do I extract new data?**
A: See [`data_extraction/README.md`](data_extraction/README.md)

**Q: Where is the trained model?**
A: `copol_prediction/artifacts/model_bundle/`

**Q: Where are the analysis plots?**
A: `copol_prediction/output/analysis/`

**Q: How do I deploy the API?**
A: See [`copol_prediction/api/README.md`](copol_prediction/api/README.md) - "Deployment" section

## 📝 Documentation Standards

All documentation in this project follows these standards:

1. **English Language**: All documentation is in English
2. **Markdown Format**: Using standard Markdown with GitHub extensions
3. **Code Examples**: Practical, runnable examples included
4. **Clear Structure**: Logical sections with clear headings
5. **Cross-References**: Links to related documentation
6. **Up-to-Date**: Documentation updated with code changes

## 🔄 Keeping Documentation Updated

When making changes:

1. **Update relevant README**: If you change functionality, update the corresponding README
2. **Update examples**: Ensure code examples still work
3. **Update cross-references**: Check links to other documentation
4. **Update version numbers**: Keep version info current
5. **Update performance metrics**: Update metrics if model changes

## 📧 Documentation Questions

If documentation is unclear or missing:

1. Check if there's a more specific README in a subdirectory
2. Look for examples in the code (`example_*.py` files)
3. Check the tests (`test_*.py` files) for usage examples
4. Open an issue on GitHub

---

**This documentation overview created**: 2025-11-14
**Last updated**: 2025-11-14
