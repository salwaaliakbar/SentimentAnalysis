# 🚀 QUICK START - 3 Minutes to Running System

## Step 1: Install (1 minute)
```bash
cd c:\Users\PMLS\Desktop\sentimentAnalysis
pip install -r requirements.txt
```

## Step 2: Verify (1 minute)
```bash
python setup_check.py
```

Expected: All ✅ checkmarks

## Step 3: Run Demo (1 minute)
```bash
python example_workflow.py
```

Expected: See complete reputation report with 10 sample comments

---

## For FYP Committee Demo

### Start API Server:
```bash
python sentiment_analysis_api.py
```

### Open Browser:
```
http://localhost:8000/docs
```

### Submit 5 Comments:
1. Use the `/submit` endpoint
2. After 5th comment → **BATCH TRIGGERS**
3. Score computed and cached

---

## Configuration

**Batch Size:** Set to 5 (in `sentiment_analysis_api.py` line 40)
```python
BATCH_THRESHOLD = 5  # For FYP demo
```

**Change for production:** Set to 20 for Glassdoor-style updates

---

## Files You Have

### Code Files (Don't delete):
- `sentiment_analyzer.py` - DistilBERT sentiment
- `aspect_extractor.py` - Extract 6 aspects
- `anti_manipulation.py` - 5-layer fraud detection
- `reputation_scorer.py` - Bayesian smoothing + CI
- `sentiment_analysis_api.py` - **Main API server**
- `example_workflow.py` - Complete demo
- `setup_check.py` - Verify setup

### Documentation:
- `COMPLETE_FYP_GUIDE.md` - **Read this for everything**
- `requirements.txt` - Python dependencies

---

## Common Commands

```bash
# Run complete demo
python example_workflow.py

# Start API server
python sentiment_analysis_api.py

# Test sentiment only
python sentiment_analyzer.py

# Test aspects only
python aspect_extractor.py

# Check setup
python setup_check.py
```

---

## Troubleshooting

**Problem:** Module not found
```bash
pip install -r requirements.txt
```

**Problem:** Port 8000 busy
```bash
# Kill process
netstat -ano | findstr :8000
# Or use different port in code
```

**Problem:** Slow performance
```bash
# In sentiment_analyzer.py, line 30:
# Change device='cpu' to device='cuda' if you have GPU
```

---

## What to Tell Committee

**System:** Corporate reputation scoring (like Glassdoor)

**Key Points:**
1. Uses DistilBERT (AI) + Bayesian smoothing (Statistics)
2. Batch processing after 5 comments (N=5 for demo)
3. Shows confidence intervals (transparency)
4. 5-layer fraud detection (rigor)
5. Per-aspect breakdown (actionable)

**Demo Flow:**
1. Start API: `python sentiment_analysis_api.py`
2. Submit 5 comments via browser (http://localhost:8000/docs)
3. Show batch trigger after 5th comment
4. Display final score with confidence + aspects

**Advantages over Glassdoor:**
- ✅ Transparent (shows methodology)
- ✅ Confident (shows CI)
- ✅ Cheap ($50-150/mo vs $1000+)
- ✅ Open-source

---

**For full details: Read `COMPLETE_FYP_GUIDE.md`**

Good luck with your presentation! 🎓✨
