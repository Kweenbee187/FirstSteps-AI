"""
=============================================================================
FIRSTSTEPS-AI — CRY CNN TRAINING PIPELINE
=============================================================================
Trains three hierarchical CNNs for infant cry analysis:

  CNN0 — ResNet18: Cry vs Non-cry gate
          Dataset : Kaggle Infant Cry Dataset (sanmithasadhish/infant-cry-dataset)
          Why     : Replaces fragile RMS/ZCR heuristic with a learned detector
          Output  : /content/cnn0_cry_detector.pth

  CNN1 — ResNet18: Hungry vs Not-hungry
          Dataset : DonateACry Corpus (github.com/gveres/donateacry-corpus)
          Output  : /content/cnn1_binary.pth

  CNN2 — ResNet18: Non-hungry subtype classifier
          Dataset : DonateACry Corpus (non-hungry samples only)
          Classes : belly_pain · discomfort · tired · burping
          Note    : layer4-only fine-tune (small dataset)
          Output  : /content/cnn2_subtype.pth
                    /content/cnn2_class_names.npy

AUDIO BRIDGE (all 3 CNNs share this):
  Raw audio → librosa Mel Spectrogram (128 mel bands, 22050 Hz, cap 10s)
            → Normalized → 224×224 RGB PNG → ResNet18

FLOW:
  Audio → CNN0 (cry?) → if yes → CNN1 (hungry?) → if no → CNN2 (subtype)
                                                         → MedGemma reasoning

SETUP (Google Colab T4):
  !pip install librosa tqdm scikit-learn torch torchvision pillow pandas kaggle

DATASET DOWNLOADS:
  # Kaggle (for CNN0):
  !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  !kaggle datasets download -d sanmithasadhish/infant-cry-dataset -p /content/kaggle_cry --unzip

  # DonateACry (for CNN1 + CNN2):
  !git clone https://github.com/gveres/donateacry-corpus.git
=============================================================================
"""

# ============================================================
# CELL 1 — Imports + device
# ============================================================
import os
import numpy as np
import pandas as pd
import librosa
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

IMG_SIZE = 224


# ============================================================
# CELL 2 — Shared: audio → mel spectrogram PNG
# ============================================================
def audio_to_spectrogram(audio_path, save_path):
    """
    Convert audio file to normalized mel spectrogram saved as PNG.
    This is the audio bridge — converts sound into a visual
    representation that ResNet18 (and MedGemma) can process.

    Settings: 128 mel bands, 22050 Hz sample rate, max 10s duration.
    """
    try:
        y, sr    = librosa.load(audio_path, sr=22050, duration=10.0)
        mel      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db   = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_img  = (mel_norm * 255).astype(np.uint8)
        img = Image.fromarray(mel_img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img.save(save_path)
        return True
    except Exception as e:
        print(f"⚠️  Failed {audio_path}: {e}")
        return False


# ============================================================
# CELL 3 — Shared: dataset class + transforms
# ============================================================
class SpectrogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx]["image_path"]).convert("RGB")
        label = int(self.df.iloc[idx]["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================
# CELL 4 — Shared: generic training loop
# ============================================================
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion,
                       epochs=10, label="Model", scheduler=None):
    """
    Generic training loop with per-epoch validation.
    Tracks best validation accuracy and prints classification report at end.
    """
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if scheduler:
            scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                preds = torch.argmax(model(images.to(device)), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        if acc > best_acc:
            best_acc = acc

        print(f"[{label}] Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.3f}")

    print(f"\n[{label}] Best Val Acc: {best_acc:.3f}")
    print(f"[{label}] Final classification report:")
    print(classification_report(all_labels, all_preds, zero_division=0))


# ============================================================
# ██████████████████████████████████████████████████████████
# CNN0 — CRY GATE (Kaggle Infant Cry Dataset)
# ██████████████████████████████████████████████████████████
# ============================================================

# ============================================================
# CELL 5 — Explore Kaggle dataset structure
# ============================================================
KAGGLE_PATH = "/content/kaggle_cry"

print("📂 Kaggle dataset structure:")
for root, dirs, files in os.walk(KAGGLE_PATH):
    level = root.replace(KAGGLE_PATH, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 2:
        for f in files[:5]:
            print(f"{indent}  {f}")

# After reading the output above, adjust CRY_FOLDER and NON_CRY_FOLDER below.
# Common structures in this Kaggle dataset:
#   /content/kaggle_cry/cry/     + /content/kaggle_cry/non_cry/
#   /content/kaggle_cry/1/       + /content/kaggle_cry/0/
#   /content/kaggle_cry/positive/+ /content/kaggle_cry/negative/


# ============================================================
# CELL 6 — Build CNN0 metadata
# ============================================================
# ---- ADJUST THESE TWO PATHS after checking Cell 5 output ----
CRY_FOLDER     = "/content/kaggle_cry/cry"
NON_CRY_FOLDER = "/content/kaggle_cry/non_cry"
# -------------------------------------------------------------

AUDIO_EXT = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
cnn0_metadata = []

for fpath in os.listdir(CRY_FOLDER):
    if fpath.lower().endswith(AUDIO_EXT):
        cnn0_metadata.append({"audio_path": os.path.join(CRY_FOLDER, fpath),
                               "label": 1, "label_name": "cry"})

for fpath in os.listdir(NON_CRY_FOLDER):
    if fpath.lower().endswith(AUDIO_EXT):
        cnn0_metadata.append({"audio_path": os.path.join(NON_CRY_FOLDER, fpath),
                               "label": 0, "label_name": "non_cry"})

df_cnn0 = pd.DataFrame(cnn0_metadata)
print(f"\nCNN0 class distribution:")
print(df_cnn0["label_name"].value_counts())
print(f"Total: {len(df_cnn0)} samples")


# ============================================================
# CELL 7 — Convert CNN0 audio → spectrograms
# ============================================================
CNN0_SPEC_PATH = "/content/spectrograms_cnn0"
os.makedirs(CNN0_SPEC_PATH, exist_ok=True)

image_paths, valid_indices = [], []

for i, row in tqdm(df_cnn0.iterrows(), total=len(df_cnn0), desc="CNN0 spectrograms"):
    fname    = os.path.splitext(os.path.basename(row["audio_path"]))[0]
    savepath = os.path.join(CNN0_SPEC_PATH, f"{fname}_{row['label']}.png")
    if audio_to_spectrogram(row["audio_path"], savepath):
        image_paths.append(savepath)
        valid_indices.append(i)

df_cnn0 = df_cnn0.loc[valid_indices].copy()
df_cnn0["image_path"] = image_paths
df_cnn0.to_csv("/content/cnn0_metadata.csv", index=False)
print(f"✅ CNN0 spectrograms done: {len(df_cnn0)} valid samples")


# ============================================================
# CELL 8 — Train CNN0
# ============================================================
df_cnn0["label"] = df_cnn0["label"].astype(int)
train_c0, val_c0 = train_test_split(df_cnn0, test_size=0.2,
                                     stratify=df_cnn0["label"], random_state=42)

# Weighted sampler — handles class imbalance
cc      = train_c0["label"].value_counts().to_dict()
total   = sum(cc.values())
sw_c0   = torch.DoubleTensor(train_c0["label"].map({k: total/v for k,v in cc.items()}).values)

train_c0_loader = DataLoader(SpectrogramDataset(train_c0, transform_train),
                             batch_size=16, sampler=WeightedRandomSampler(sw_c0, len(sw_c0)))
val_c0_loader   = DataLoader(SpectrogramDataset(val_c0, transform_val),
                             batch_size=16, shuffle=False)

# ResNet18 — full fine-tune (Kaggle dataset is large enough)
model0 = models.resnet18(pretrained=True)
model0.fc = nn.Linear(model0.fc.in_features, 2)
model0 = model0.to(device)

optimizer0 = optim.Adam(model0.parameters(), lr=1e-4)
scheduler0 = optim.lr_scheduler.StepLR(optimizer0, step_size=5, gamma=0.5)

print("🟣 Training CNN0 — Cry Gate...")
train_and_evaluate(model0, train_c0_loader, val_c0_loader,
                   optimizer0, nn.CrossEntropyLoss(),
                   epochs=12, label="CNN0-CryGate", scheduler=scheduler0)

torch.save(model0.state_dict(), "/content/cnn0_cry_detector.pth")
print("✅ CNN0 saved → /content/cnn0_cry_detector.pth")


# ============================================================
# ██████████████████████████████████████████████████████████
# CNN1 + CNN2 — CRY TYPE (DonateACry Corpus)
# ██████████████████████████████████████████████████████████
# ============================================================

# ============================================================
# CELL 9 — Explore DonateACry corpus
# ============================================================
BASE_AUDIO_PATH = "/content/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"

print("📊 DonateACry file count per tag:")
for folder in sorted(os.listdir(BASE_AUDIO_PATH)):
    folder_path = os.path.join(BASE_AUDIO_PATH, folder)
    if os.path.isdir(folder_path):
        count = sum(1 for _, _, files in os.walk(folder_path) for f in files if f.endswith('.wav'))
        print(f"  {folder}: {count} files")

# Label codes embedded in filenames:
#   hu = hungry | bp = belly_pain | bu = burping | dc = discomfort | ti = tired


# ============================================================
# CELL 10 — Convert DonateACry audio → spectrograms
# ============================================================
OUTPUT_SPEC_PATH = "/content/spectrograms"
os.makedirs(OUTPUT_SPEC_PATH, exist_ok=True)

metadata = []

for folder in os.listdir(BASE_AUDIO_PATH):
    folder_path = os.path.join(BASE_AUDIO_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    save_folder = os.path.join(OUTPUT_SPEC_PATH, folder)
    os.makedirs(save_folder, exist_ok=True)

    for file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
        if not file.lower().endswith(".wav"):
            continue
        audio_path = os.path.join(folder_path, file)
        label_code = file.split("-")[-1].replace(".wav", "").strip()
        save_path  = os.path.join(save_folder, file.replace(".wav", ".png"))

        if audio_to_spectrogram(audio_path, save_path):
            metadata.append({
                "audio_path":     audio_path,
                "image_path":     save_path,
                "folder_label":   folder,
                "filename_label": label_code
            })

print(f"\n✅ DonateACry spectrograms done: {len(metadata)} files")


# ============================================================
# CELL 11 — Build DonateACry metadata + decode labels
# ============================================================
df = pd.DataFrame(metadata)

CODE_MAP = {"hu": "hungry", "bp": "belly_pain", "bu": "burping",
            "dc": "discomfort", "ti": "tired"}

df["decoded_label"] = df["filename_label"].map(CODE_MAP)
df = df[df["decoded_label"].notna()].copy()
df["binary_label"] = df["decoded_label"].apply(
    lambda x: "hungry" if x == "hungry" else "not_hungry"
)

df.to_csv("/content/cry_metadata.csv", index=False)

print("Label distribution:")
print(df["decoded_label"].value_counts())
print(f"\n⚠️  Note: dataset is ~83% hungry — weighted sampling handles this")


# ============================================================
# CELL 12 — Train CNN1: Hungry vs Not-hungry
# ============================================================
df_binary = df.copy()
df_binary["label"] = df_binary["decoded_label"].apply(lambda x: 1 if x == "hungry" else 0)

train_b, val_b = train_test_split(df_binary, test_size=0.2,
                                   stratify=df_binary["label"], random_state=42)

cc_b  = train_b["label"].value_counts().to_dict()
tot_b = sum(cc_b.values())
sw_b  = torch.DoubleTensor(train_b["label"].map({k: tot_b/v for k,v in cc_b.items()}).values)

train_b_loader = DataLoader(SpectrogramDataset(train_b, transform_train),
                            batch_size=16, sampler=WeightedRandomSampler(sw_b, len(sw_b)))
val_b_loader   = DataLoader(SpectrogramDataset(val_b, transform_val),
                            batch_size=16, shuffle=False)

model1 = models.resnet18(pretrained=True)
model1.fc = nn.Linear(model1.fc.in_features, 2)
model1 = model1.to(device)

optimizer1 = optim.Adam(model1.parameters(), lr=1e-4)

print("🔵 Training CNN1 — Hungry vs Not-hungry...")
train_and_evaluate(model1, train_b_loader, val_b_loader,
                   optimizer1, nn.CrossEntropyLoss(),
                   epochs=10, label="CNN1-Binary")

torch.save(model1.state_dict(), "/content/cnn1_binary.pth")
print("✅ CNN1 saved → /content/cnn1_binary.pth")


# ============================================================
# CELL 13 — Train CNN2: Non-hungry subtype classifier
# ============================================================
df_sub = df[df["decoded_label"] != "hungry"].copy()
df_sub["decoded_label"] = df_sub["decoded_label"].str.strip().str.lower()
df_sub["label"], class_names = pd.factorize(df_sub["decoded_label"])

# Save class names — required for interface to load CNN2 correctly
np.save("/content/cnn2_class_names.npy", class_names)
print(f"CNN2 classes: {list(class_names)}")
print(df_sub["decoded_label"].value_counts())

train_s, val_s = train_test_split(df_sub, test_size=0.2,
                                   stratify=df_sub["label"], random_state=42)

cc_s  = train_s["label"].value_counts().to_dict()
tot_s = sum(cc_s.values())
sw_s  = torch.DoubleTensor(train_s["label"].map({k: tot_s/v for k,v in cc_s.items()}).values)

train_s_loader = DataLoader(SpectrogramDataset(train_s, transform_train),
                            batch_size=16, sampler=WeightedRandomSampler(sw_s, len(sw_s)))
val_s_loader   = DataLoader(SpectrogramDataset(val_s, transform_val),
                            batch_size=16, shuffle=False)

# Layer4-only fine-tune — dataset is small (non-hungry samples only)
# Freezing early layers prevents overfitting
model2 = models.resnet18(pretrained=True)
for param in model2.parameters():
    param.requires_grad = False
for param in model2.layer4.parameters():
    param.requires_grad = True

num_classes = len(class_names)
model2.fc = nn.Linear(model2.fc.in_features, num_classes)
model2 = model2.to(device)

# Class-weighted loss for imbalanced subtype distribution
cw_s      = torch.tensor([tot_s / cc_s[i] for i in range(num_classes)],
                          dtype=torch.float32).to(device)
criterion2 = nn.CrossEntropyLoss(weight=cw_s)
optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=1e-4)

print("🟠 Training CNN2 — Subtype classifier...")
train_and_evaluate(model2, train_s_loader, val_s_loader,
                   optimizer2, criterion2,
                   epochs=15, label="CNN2-Subtype")

torch.save(model2.state_dict(), "/content/cnn2_subtype.pth")
print("✅ CNN2 saved → /content/cnn2_subtype.pth")


# ============================================================
# CELL 14 — Quick inference test on all 3 CNNs
# ============================================================
def hierarchical_predict(audio_path):
    """
    Run full 3-stage pipeline on a single audio file.
    Returns dict with prediction + all probabilities.
    """
    spec_path = "/content/temp_test_spec.png"
    audio_to_spectrogram(audio_path, spec_path)
    tensor = transform_val(Image.open(spec_path).convert("RGB")).unsqueeze(0).to(device)

    # Stage 0 — Cry gate
    model0.eval()
    with torch.no_grad():
        p0 = F.softmax(model0(tensor), dim=1).cpu().numpy()[0]
    if float(p0[1]) <= 0.5:
        return {"is_cry": False, "cry_prob": float(p0[1]), "prediction": "non_cry"}

    # Stage 1 — Hungry?
    model1.eval()
    with torch.no_grad():
        p1 = F.softmax(model1(tensor), dim=1).cpu().numpy()[0]
    hungry_p, not_hungry_p = float(p1[1]), float(p1[0])

    if hungry_p > 0.5:
        return {"is_cry": True, "cry_prob": float(p0[1]),
                "prediction": "hungry", "confidence": hungry_p}

    # Stage 2 — Subtype
    class_names_loaded = np.load("/content/cnn2_class_names.npy", allow_pickle=True)
    model2.eval()
    with torch.no_grad():
        p2 = F.softmax(model2(tensor), dim=1).cpu().numpy()[0]

    subtype_probs = {class_names_loaded[i]: float(p2[i]) * not_hungry_p
                     for i in range(len(class_names_loaded))}
    prediction    = max(subtype_probs, key=subtype_probs.get)

    return {
        "is_cry": True,
        "cry_prob": float(p0[1]),
        "prediction": prediction,
        "probs": {"hungry": hungry_p, **subtype_probs}
    }


# Test on first sample
test_audio = df.iloc[0]["audio_path"]
result     = hierarchical_predict(test_audio)
print("\n🧪 Inference test:")
print(f"  File      : {os.path.basename(test_audio)}")
print(f"  Is cry    : {result['is_cry']}")
print(f"  Cry prob  : {result['cry_prob']:.3f}")
print(f"  Prediction: {result['prediction']}")


# ============================================================
# CELL 15 — Evaluate full pipeline on DonateACry held-out set
# ============================================================
from sklearn.metrics import classification_report as cr

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    try:
        res = hierarchical_predict(row["audio_path"])
        results.append({
            "gold":     row["decoded_label"],
            "pred":     res["prediction"],
            "is_cry":   res["is_cry"],
            "cry_prob": res["cry_prob"]
        })
    except Exception as e:
        print(f"⚠️  {e}")

results_df = pd.DataFrame(results)
results_df.to_csv("/content/full_pipeline_results.csv", index=False)

print("\n📊 PIPELINE EVALUATION")
print(f"Cry detection rate: {results_df['is_cry'].mean():.3f}")
print("\n5-class report:")
print(cr(results_df["gold"], results_df["pred"], zero_division=0))


# ============================================================
# CELL 16 — Summary
# ============================================================
print("""
✅ ALL CRY MODELS SAVED:

  /content/cnn0_cry_detector.pth   — CNN0: Cry Gate (Kaggle-trained)
  /content/cnn1_binary.pth         — CNN1: Hungry vs Not-hungry
  /content/cnn2_subtype.pth        — CNN2: belly_pain / discomfort / tired / burping
  /content/cnn2_class_names.npy    — Class ordering for CNN2 (required by interface)

⚠️  Download all 4 files to Google Drive before Colab resets.

Next step: Run skin_cnn_training.py to train CNN3, then launch the interface.
""")
