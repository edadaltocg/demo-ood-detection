"""
Gradio demo of image classification with OOD detection.

If the image example is probably OOD, the model will abstain from the prediction.
"""
import json
import logging
import pickle
from glob import glob

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn.functional as F
from gradio.components import JSON, Image, Label
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

_logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 3

# load model
print("Loading model...")
model = timm.create_model("resnet50.tv2_in1k", pretrained=True, checkpoint_path="resnet50.tv2_in1k.bin")
model.to(device)
model.eval()

# dataset labels
idx2label = json.loads(open("ilsvrc2012.json").read())
idx2label = {int(k): v for k, v in idx2label.items()}
print(idx2label)
print(idx2label.values())

# transformation
config = resolve_data_config({}, model=model)
config["is_training"] = False
transform = create_transform(**config)

# create feature extractor
penultimate_features_key = "global_pool.flatten"
logits_key = "fc"
features_names = [penultimate_features_key, logits_key]

feature_extractor = create_feature_extractor(model, features_names)

centroids = torch.from_numpy(pickle.load(open("centroids_resnet50.tv2_in1k_igeood_logits.pkl", "rb"))).to(device)
# OOD detector thresholds
msp_threshold = 0.3796
energy_threshold = 0.3781
igeood_threshold = 2.4984


def mahalanobis_penult(features):
    scores = torch.norm(features, dim=1, keepdims=True)
    s = torch.min(scores, dim=1)[0]
    return -s.item()


def msp(logits):
    return torch.softmax(logits, dim=1).max(-1)[0].item()


def energy(logits):
    return torch.logsumexp(logits, dim=1).item()


def igeoodlogits_vec(logits, temperature, centroids, epsilon=1e-12):
    logits = torch.sqrt(F.softmax(logits / temperature, dim=1))
    centroids = torch.sqrt(F.softmax(centroids / temperature, dim=1))
    mult = logits @ centroids.T
    stack = 2 * torch.acos(torch.clamp(mult, -1 + epsilon, 1 - epsilon))
    return stack.mean(dim=1).item()


def predict(image):
    # forward pass
    inputs = transform(image).unsqueeze(0)
    inputs = inputs.to(device)
    with torch.no_grad():
        features = feature_extractor(inputs)

    # top 5 predictions
    probabilities = torch.softmax(features[logits_key], dim=-1)
    softmax, class_idxs = torch.topk(probabilities, TOPK)
    _logger.info(softmax)
    _logger.info(class_idxs)

    result = {idx2label[i.item()]: v.item() for i, v in zip(class_idxs.squeeze(), softmax.squeeze())}
    # OOD
    msp_score = round(msp(features[logits_key]), 4)
    energy_score = round(energy(features[logits_key]), 4)
    igeood_scores = round(igeoodlogits_vec(features[logits_key], 1, centroids), 4)
    ood_scores = {
        "MSP": msp_score,
        "MSP, is the input OOD?": msp_score < msp_threshold,
        "Energy": energy_score,
        "Energy, is the input OOD?": energy_score < energy_threshold,
        "Igeood": igeood_scores,
        "Igeood, is the input OOD?": igeood_scores < igeood_threshold,
    }
    _logger.info(ood_scores)
    return result, ood_scores


def main():
    # image examples for demo shuffled
    examples = glob("images/imagenet/*") + glob("images/ood/*")
    np.random.seed(42)
    # np.random.shuffle(examples)

    # gradio interface
    interface = gr.Interface(
        fn=predict,
        inputs=Image(type="pil"),
        outputs=[
            Label(num_top_classes=TOPK, label="Model prediction"),
            JSON(label="OOD scores"),
        ],
        examples=examples,
        examples_per_page=len(examples),
        allow_flagging="never",
        theme="default",
        title="OOD Detection 🧐",
        description=(
            "Out-of-distribution (OOD) detection is an essential safety measure for machine learning models. "
            "The objective of an OOD detector is to determine wether the input sample comes from the distribution known by the AI model. "
            "For instance, an input that does not belong to any of the known classes or is from a different domain should be flagged by the detector.\n"
            "In this demo we will display the decision of three OOD detectors on a ResNet-50 model trained to classify on the ImageNet-1K dataset (top-1 accuracy 80%)."
            "This model can classify among 1000 classes from several categories, including `animals`, `vehicles`, `clothing`, `instruments`, `plants`, etc. "
            "For the complete hierarchy of classes, please check the website https://observablehq.com/@mbostock/imagenet-hierarchy. "
            "\n\n"
            "## Instructions:\n"
            "1. Upload an image of your choice or select one from the examples bar.\n"
            "2. The model will predict the top 3 most likely classes for the image.\n"
            "3. The OOD detectors will output their scores and decision on the image. The smaller the score, the least confident the detector is on the sample being in-distribution.\n"
            "4. If the image is OOD, the model will abstain from the prediction and flag it to the practicioner.\n"
            "\n\n\nEnjoy the demo!"
        ),
        cache_examples=True,
    )
    interface.launch(server_port=7860)
    interface.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)

    gr.close_all()
    main()
