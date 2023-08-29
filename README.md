---
title: OOD Detection
emoji: 🧐
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 3.12.0
app_file: app.py
pinned: true
license: mit
---

# OOD Detection Demo 🧐

Out-of-distribution (OOD) detection is an essential safety measure for machine learning models. This app demonstrates how these methods can be useful. They try to determine wether we can trust the predictions of a ResNet-50 model trained on ImageNet-1K.

This demo is [online](https://huggingface.co/spaces/edadaltocg/ood-detection) at `https://huggingface.co/spaces/edadaltocg/ood-detection`

## Running Gradio app locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open the app in your browser at `http://localhost:7860`.
