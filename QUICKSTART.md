# 🚀 QUICK START - 3 Minutes to Running System

## Step 1: Install (1 minute)
```bash
cd c:\Users\PMLS\Desktop\sentimentAnalysis
pip install -r requirements.txt
```

## Step 2: Place Trained Model (1 minute)
Copy your `model_output` folder (from Google Drive) into the project root:

```
SentimentAnalysis/
	model_output/
		best.pt
		tokenizer_config.json
		vocab.txt
		...
```

## Step 3: Verify (1 minute)
```bash
python setup_check.py
```

Expected: All ✅ checkmarks

## Step 4: Run Demo (1 minute)
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

### Analyze Comments:
1. Use the `/analyze` endpoint
2. Provide a list of comments
3. Get sentiment scores back for each comment

---

## Files You Have

### Code Files (Don't delete):
- `sentiment_analyzer.py` - DistilBERT sentiment
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
2. Multitask outputs overall + aspect scores
3. 5-layer fraud detection (rigor)
4. Per-aspect breakdown (actionable)

**Demo Flow:**
1. Start API: `python sentiment_analysis_api.py`
2. Submit comments via browser (http://localhost:8000/docs)
3. Display sentiment scores + aspect outputs

**Advantages over Glassdoor:**
- ✅ Transparent (shows methodology)
- ✅ Confident (shows CI)
- ✅ Cheap ($50-150/mo vs $1000+)
- ✅ Open-source

---

**For full details: Read `COMPLETE_FYP_GUIDE.md`**

Good luck with your presentation! 🎓✨
