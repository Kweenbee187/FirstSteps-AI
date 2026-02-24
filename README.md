# 🍼 FirstSteps-AI
### *Your Infant's First Voice. Decoded.*

**AI Nurse & Dermatologist for First-Time Parents**  
Powered by MedGemma · Kaggle MedGemma Impact Challenge 2025  
`Main Track` · `Novel Task Track` · `MedGemma-1.5-4b-it`

> *"It's 3 AM. Your newborn won't stop crying. Priya and Arjun are first-time parents. Their 6-week-old has been crying for 40 minutes. Is she hungry? In pain? Is that red patch on her arm something serious? They Google frantically — 15 browser tabs, contradictory advice, medical jargon, no real answers. Just more panic. Their pediatrician's office opens in 6 hours.*
> 
> *Every first-time parent deserves a knowledgeable, calm voice at 3 AM. FirstSteps-AI is that voice."*

---

## 🧁 Team Muffin

| Name | GitHub |
|------|--------|
| Sneha Chakraborty | [@Kweenbee187](https://github.com/Kweenbee187) |
| Divyansh Pathak | [@tituatgithub](https://github.com/tituatgithub) |

---

## What is FirstSteps-AI?

FirstSteps-AI is a multimodal infant health assistant with two AI-powered modes and one consistent voice — a warm, knowledgeable pediatric nurse talking directly to anxious first-time parents. Never clinical jargon. Never generic advice. Always specific, always actionable.

| Mode | Input | Pipeline | Output |
|------|-------|----------|--------|
| 🍼 **AI Nurse** | Baby cry audio | CNN0 → CNN1 → CNN2 → MedGemma | Condition-specific nurse guidance |
| 🔬 **AI Dermatologist** | Skin photo | CNN3 → MedGemma (sees image) | Dermatologist guidance with visual confirmation |

---

## Architecture

```
🍼 CRY PIPELINE
───────────────────────────────────────────────────────────────
Audio Input (WAV/MP3/OGG/M4A)
    │
    ▼
[Audio → Mel Spectrogram] ← librosa, 128 mel bands, 22050 Hz, 224×224 RGB PNG
    │
    ▼
CNN0 — Cry Gate (ResNet18, Kaggle-trained)
    │  cry_prob > 0.5?
    ├── NO  → "Not a cry" — stop here
    └── YES ↓
CNN1 — Hungry? (ResNet18, DonateACry)
    │  hungry_prob > 0.5?
    ├── YES → "Hungry" → MedGemma
    └── NO  ↓
CNN2 — Subtype (ResNet18, DonateACry, layer4 fine-tune)
    │  4 classes: belly_pain / discomfort / tired / burping
    └──→ MedGemma
            │  Inputs: waveform image (vision) + acoustic features (text) + CNN probs (text)
            └──→ Structured nurse guidance (6 sections)

🔬 SKIN PIPELINE
───────────────────────────────────────────────────────────────
Image Input (JPG/PNG)
    │
    ▼
CNN3 — Skin Classifier (ResNet18, DermNet, 7 conditions)
    │
    └──→ MedGemma (receives actual skin photo + CNN probabilities)
            └──→ Structured dermatologist guidance (6 sections)
```

### Key Novelty — Audio→Vision→LLM Bridge

MedGemma is an image-text model with **no native audio understanding**. We made it reason about sound by:

1. Converting raw cry audio → Mel spectrogram image (128 mel bands, 22050 Hz, normalized to 224×224 RGB PNG)
2. Feeding that image through MedGemma's **vision encoder**
3. Injecting acoustic feature vectors (RMS, ZCR, MFCC, Spectral Centroid) as clinical evidence in the text prompt
4. Fusing CNN softmax probabilities to shape MedGemma's confidence framing and advice tone

This is a **triple-signal fusion prompt**: waveform image (vision) + acoustic features (text) + CNN probability distribution (text) → one structured clinical response.

---

## Models

| Model | Architecture | Dataset | Task | Training Details |
|-------|-------------|---------|------|-----------------|
| **CNN0** | ResNet18, full fine-tune | Kaggle Infant Cry Dataset | Cry vs Non-cry (binary) | 12 epochs, StepLR (step=5, γ=0.5), WeightedRandomSampler |
| **CNN1** | ResNet18, full fine-tune | DonateACry Corpus | Hungry vs Not-hungry (binary) | 10 epochs, Adam lr=1e-4, WeightedRandomSampler |
| **CNN2** | ResNet18, layer4-only fine-tune | DonateACry Corpus (non-hungry) | Subtype: belly_pain / discomfort / tired / burping | 15 epochs, class-weighted CrossEntropy |
| **CNN3** | ResNet18, full fine-tune | DermNet Dataset (7 classes) | Skin condition classification | 10 epochs, Adam lr=1e-4, F1-macro as primary metric |
| **MedGemma** | `google/medgemma-1.5-4b-it` | — | Clinical reasoning (both pipelines) | Inference only, bfloat16, GPU |

---

## Datasets

### CNN0 — Cry Gate
- **Dataset:** [Infant Cry Dataset](https://www.kaggle.com/datasets/sanmithasadhish/infant-cry-dataset) (Kaggle, sanmithasadhish)
- **Classes:** `cry` (label 1) · `non_cry` (label 0)
- **Why:** Replaced a fragile RMS/ZCR heuristic gate with a learned binary detector trained on real labeled cry/non-cry audio

### CNN1 + CNN2 — Cry Type
- **Dataset:** [DonateACry Corpus](https://github.com/gveres/donateacry-corpus) (457 samples)
- **Label codes in filenames:** `hu` = hungry · `bp` = belly_pain · `bu` = burping · `dc` = discomfort · `ti` = tired
- **Class imbalance:** ~83% hungry — handled with WeightedRandomSampler (CNN1) and class-weighted CrossEntropy (CNN2)
- **Audio bridge:** All audio converted to 128-band Mel spectrograms at 22050 Hz, normalized, saved as 224×224 RGB PNG

### CNN3 — Skin Classifier
- **Dataset:** [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) (Kaggle)
- **7 infant-relevant classes selected from DermNet train folder:**

```
Atopic Dermatitis Photos
Cellulitis Impetigo and other Bacterial Infections
Exanthems and Drug Eruptions
Tinea Ringworm Candidiasis and other Fungal Infections
Scabies Lyme Disease and other Infestations and Bites
Warts Molluscum and other Viral Infections
Urticaria Hives
```

- **Selection rationale:** All 7 conditions occur in infants and young children, can be photographed at home, and benefit from early parental recognition

---

## MedGemma Prompt Engineering

Both pipelines use strict structured prompts that force:

1. **Condition-specific output only** — no generic baby advice, no listing other conditions as alternatives
2. **Named treatments** — specific ingredient names (clotrimazole 1%, simethicone drops, zinc oxide) not vague "use OTC cream"
3. **Concrete thresholds** — exact temperatures (38°C / 100.4°F), durations (hours of crying, days without improvement), spreading distances (cm)
4. **Warm nurse tone** — written as a knowledgeable nurse talking directly to an anxious first-time parent
5. **One warm disclaimer** — always the last line, never multiple legal paragraphs

### Cry Mode — 6 Output Sections
```
Condition_Overview     → What is happening in the baby's body + what the acoustic pattern indicates
Immediate_Actions      → Step-by-step what to do in the next 5-10 minutes
Treatment_and_Relief   → Named techniques, holds, or specific OTC products
Warning_Signs          → Exact symptoms that mean this has become more serious
When_to_Call_Doctor    → Concrete thresholds: temperature, hours, behaviors
Disclaimer             → One warm sentence
```

### Skin Mode — 6 Output Sections
```
Condition_Overview     → What this condition is, how common in babies, urgent or manageable
What_You_Might_See     → Visual appearance description referencing the uploaded image
Immediate_Care         → Today's hygiene, clothing, environment steps
Safe_Treatments        → Named ingredients safe for infants + how to apply + what to avoid
Warning_Signs          → Specific visual/behavioral changes
When_to_Call_Doctor    → Fever °C/°F, spread cm, days, secondary infection signs
Disclaimer             → One warm sentence
```

---

## Interface

The Gradio interface (`infant_cry_interface.py`) runs in Google Colab and auto-detects available models at startup.

**Three modes (auto-detected):**
- 🔵 `DIAGNOSTIC` — No weights found → acoustic plots only
- 🟡 `CNN MODE` — CNNs loaded, MedGemma not available
- 🟢 `FULL MODE` — Everything loaded

**UI:** Two mode toggle buttons (`🍼 Cry Analysis` / `🔬 Skin Analysis`) switch panels without page reload. All CNNs run on CPU, MedGemma runs on GPU (T4).

---

## File Structure

```
FirstSteps-AI/
├── README.md
├── LICENSE                          ← MIT
├── FirstSteps-AI.pdf                ← Competition slides
├── infant_cry_analysis_main.py      ← Full CNN0+1+2+3 training pipeline
├── infant_cry_interface.py          ← Gradio interface (two-mode toggle UI)
└── notebooks/
    └── FirstSteps_AI.ipynb          ← Original Colab notebook
```

---

## Setup & Running

### 1. Install dependencies
```bash
pip install gradio librosa matplotlib torchvision transformers accelerate nest_asyncio kaggle
```

### 2. HuggingFace token (for MedGemma)
```python
from huggingface_hub import login
login("your_hf_token")  # needs access to google/medgemma-1.5-4b-it
```

### 3. Kaggle credentials (for CNN0 dataset)
```python
# Upload kaggle.json in Colab
from google.colab import files
files.upload()

import os
os.makedirs("~/.kaggle", exist_ok=True)
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 4. Training (run `infant_cry_analysis_main.py` cells in order)

| Cell | What it does |
|------|-------------|
| 1 | Install dependencies |
| 2 | Download Kaggle infant cry dataset |
| 3 | Explore dataset structure |
| 4 | Build CNN0 metadata (cry/non-cry) |
| 5 | Convert CNN0 audio → spectrograms |
| 6 | PyTorch setup + shared dataset class |
| 7 | **Train CNN0** — Cry gate |
| 8 | Clone DonateACry corpus |
| 9 | Convert DonateACry audio → spectrograms |
| 10 | Build DonateACry metadata + label decoding |
| 11 | **Train CNN1** — Hungry/Not-hungry |
| 12 | **Train CNN2** — Subtype classifier |
| 13 | Test CNN0 on a sample |
| 14 | Full hierarchical inference test |
| 15 | Evaluate full pipeline |
| 16 | Summary of saved model paths |

**CNN3 (skin) is trained separately** — run the skin training cells with DermNet dataset mounted at `/content/dermnet/train/`.

### 5. Launch interface
```python
# After training, run infant_cry_interface.py
# Models expected at:
# /content/cnn0_cry_detector.pth
# /content/cnn1_binary.pth
# /content/cnn2_subtype.pth
# /content/cnn2_class_names.npy
# /content/cnn3_skin_classifier.pth
# /content/cnn3_class_names.npy   ← save this after CNN3 training:
import numpy as np
np.save("/content/cnn3_class_names.npy", class_names_cnn3)
```

---

## Results

**Cry pipeline performance** (evaluated on DonateACry held-out set):
- CNN0 replaces a fragile heuristic gate with a learned detector — eliminates false positives from non-cry audio
- CNN1 + CNN2 hierarchical cascade handles the severe class imbalance (83% hungry) via weighted sampling at each stage
- MedGemma outputs verified to be condition-specific, not generic, with concrete clinical thresholds

**Skin pipeline:**
- CNN3 trained with F1-macro as primary metric (not accuracy) to handle DermNet class imbalance
- MedGemma receives the actual skin image — enables genuine visual grounding in the `What_You_Might_See` section, not just text generation

---

## Why This Matters

- **140M+** babies born globally each year
- **3 AM** is when parents need answers most — clinics are closed, anxiety is highest
- **15 browser tabs** of contradictory medical jargon is what parents currently get
- FirstSteps-AI gives them one calm, specific, actionable voice — always available, always free

**Responsible AI:** Every response ends with a warm one-sentence disclaimer. The tool is designed to assist, never replace, pediatricians. It explicitly directs parents to call their doctor with concrete criteria for when "now" means now.

---

## Competition Tracks

**🏆 Main Track — Real Healthcare Impact**
- MedGemma-1.5-4b-it at the core of all clinical reasoning
- Structured multimodal prompts, not decoration
- Warm pediatric nurse tone enforced through prompt engineering

**🚀 Novel Task Track — Breaking MedGemma's Modality Limit**
- MedGemma is image/text only — no native audio support
- We made it reason about sound via the Audio→Vision→LLM bridge
- First-of-its-kind pipeline: cry audio → mel spectrogram → MedGemma vision encoder → clinical guidance
- Triple-signal fusion: waveform image + acoustic feature vectors + CNN softmax distribution

---

## License

MIT License — see [LICENSE](LICENSE)

---

*FirstSteps-AI starts as a competition entry. It ends as infrastructure for infant healthcare.*

**🧁 Team Muffin · Kaggle MedGemma Impact Challenge 2025**
