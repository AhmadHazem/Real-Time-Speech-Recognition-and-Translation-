import torch
import torchaudio
from datasets import load_dataset, load_metric
from lang_trans.arabic import buckwalter
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

# Load the Arabic Common Voice dataset
test_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ar", split="test")
wer = load_metric("wer")

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("Zaid/wav2vec2-large-xlsr-53-arabic-egyptian")
model = Wav2Vec2ForCTC.from_pretrained("Zaid/wav2vec2-large-xlsr-53-arabic-egyptian")
model.to("cuda")

# Characters to ignore during preprocessing
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“]'
resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Evaluate the model
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

# Compute and print the Word Error Rate (WER)
print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))
