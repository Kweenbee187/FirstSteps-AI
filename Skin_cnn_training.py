"""
=============================================================================
FIRSTSTEPS-AI — SKIN CNN TRAINING PIPELINE
=============================================================================
Trains CNN3 for infant skin condition classification:

  CNN3 — ResNet18: 7-class skin condition classifier
          Dataset : DermNet (kaggle.com/datasets/shubhamgoel27/dermnet)
          Classes : 7 infant-relevant conditions selected from DermNet train/
          Metric  : F1-macro (primary) — accuracy alone misleads on imbalanced data
          Output  : /content/cnn3_skin_classifier.pth
                    /content/cnn3_class_names.npy

WHY THESE 7 CONDITIONS:
  All occur in infants and young children, can be photographed at home,
  and benefit from early parental recognition. Selected from DermNet's
  full label set specifically for pediatric relevance.

  Atopic Dermatitis Photos
  Cellulitis Impetigo and other Bacterial Infections
  Exanthems and Drug Eruptions
  Tinea Ringworm Candidiasis and other Fungal Infections
  Scabies Lyme Disease and other Infestations and Bites
  Warts Molluscum and other Viral Infections
  Urticaria Hives

IN THE INTERFACE:
  CNN3 output (softmax probabilities) + original skin photo both get
  passed to MedGemma — enabling genuine visual grounding in the
  What_You_Might_See section, not just text generation from class labels.

SETUP (Google Colab T4):
  !pip install torch torchvision scikit-learn pillow pandas kaggle

DATASET DOWNLOAD:
  !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  !kaggle datasets download -d shubhamgoel27/dermnet -p /content/dermnet --unzip
=============================================================================
"""

# ============================================================
# CELL 1 — Imports + device
# ============================================================
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ============================================================
# CELL 2 — Build CNN3 metadata from DermNet
# ============================================================
DERMNET_PATH = "/content/dermnet/train"

TARGET_CLASSES = [
    "Atopic Dermatitis Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Exanthems and Drug Eruptions",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Warts Molluscum and other Viral Infections",
    "Urticaria Hives"
]

metadata = []

for folder in TARGET_CLASSES:
    folder_path = os.path.join(DERMNET_PATH, folder)
    if not os.path.isdir(folder_path):
        print(f"⚠️  Folder not found: {folder_path}")
        continue
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            metadata.append({
                "image_path": os.path.join(folder_path, img_file),
                "label_name": folder
            })

df_cnn3 = pd.DataFrame(metadata)
df_cnn3["label"], class_names_cnn3 = pd.factorize(df_cnn3["label_name"])

print("✅ CNN3 Classes:")
for i, name in enumerate(class_names_cnn3):
    print(f"  {i} → {name}")

print(f"\n📊 Samples per class:")
print(df_cnn3["label_name"].value_counts())
print(f"\nTotal: {len(df_cnn3)} samples")

# Save class names immediately — required for interface to load CNN3 correctly
np.save("/content/cnn3_class_names.npy", class_names_cnn3)
print("\n✅ Class names saved → /content/cnn3_class_names.npy")


# ============================================================
# CELL 3 — Dataset class + transforms
# ============================================================
class SkinDataset(Dataset):
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
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ============================================================
# CELL 4 — Train / val split
# ============================================================
train_df, val_df = train_test_split(
    df_cnn3, test_size=0.2, stratify=df_cnn3["label"], random_state=42
)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")

train_loader = DataLoader(SkinDataset(train_df, transform_train),
                          batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(SkinDataset(val_df, transform_val),
                          batch_size=16, shuffle=False, num_workers=2)


# ============================================================
# CELL 5 — Model + class-weighted loss
# ============================================================
model3 = models.resnet18(pretrained=True)
model3.fc = nn.Linear(model3.fc.in_features, len(class_names_cnn3))
model3 = model3.to(device)

# Class weights — critical for DermNet imbalance
# Inverse frequency: rarer classes get higher weight
class_counts = train_df["label"].value_counts().sort_index()
total        = class_counts.sum()
weights      = torch.tensor([total / c for c in class_counts], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model3.parameters(), lr=1e-4)

print("Class weights applied:")
for i, (name, w) in enumerate(zip(class_names_cnn3, weights.cpu())):
    short = name.split(" Photos")[0].split(" and other")[0][:40]
    print(f"  {i} {short}: {w:.2f}")


# ============================================================
# CELL 6 — Training loop (F1-macro tracked, best model saved)
# ============================================================
EPOCHS  = 10
best_f1 = 0.0

print("\n🟣 Training CNN3 — Skin Classifier...")

for epoch in range(EPOCHS):

    # ── Train ──
    model3.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model3(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ── Validate ──
    model3.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            preds = torch.argmax(model3(images.to(device)), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    import numpy as np
    val_f1  = f1_score(all_labels, all_preds, average="macro")
    val_acc = (np.array(all_preds) == np.array(all_labels)).mean()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss     : {total_loss/len(train_loader):.4f}")
    print(f"  Val Accuracy   : {val_acc:.4f}")
    print(f"  Val F1 (Macro) : {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model3.state_dict(), "/content/cnn3_skin_classifier.pth")
        print("  ✅ Best model saved")

print(f"\n🔥 Training Complete | Best F1: {best_f1:.4f}")


# ============================================================
# CELL 7 — Final classification report
# ============================================================
model3.load_state_dict(torch.load("/content/cnn3_skin_classifier.pth"))
model3.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        preds = torch.argmax(model3(images.to(device)), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

short_names = [n.split(" Photos")[0].split(" and other")[0][:35]
               for n in class_names_cnn3]

print("\n📊 FINAL CLASSIFICATION REPORT:")
print(classification_report(all_labels, all_preds,
                            target_names=short_names, zero_division=0))

print("""
✅ CNN3 SAVED:

  /content/cnn3_skin_classifier.pth  — Model weights (best F1-macro)
  /content/cnn3_class_names.npy      — Class ordering (required by interface)

⚠️  Download both files to Google Drive before Colab resets.

Next step: Launch infant_cry_interface.py with all 4 CNN weights loaded.
""")
