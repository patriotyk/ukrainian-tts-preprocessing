# Ukrainian stress prediction using ByT5 and CTranslate2

This is a Python library for prediction stress in Ukrainian words by model [mouseyy/stressifier-byt5-g2p-model](https://huggingface.co/mouseyy/stressifier-byt5-g2p-model) converted to CTranslate2 format.

Preprocess and postprocess code has been adapted from [lang-uk/ukrainian-tts-preprocessing](https://github.com/lang-uk/ukrainian-tts-preprocessing) repository.

Comparing with original model [mouseyy/stressifier-byt5-g2p-model](https://huggingface.co/mouseyy/stressifier-byt5-g2p-model) it works 3 times faster on GPU, and depends only on ctranslate2 and huggingface_hub libraries.


# Installation

```bash
pip install https://github.com/patriotyk/stressifier-byt5-ctranslate2
```

# How to use

```python
from ukrainian_stressifier_byt5_ct2 import UkrainianStressifier

stressifier = UkrainianStressifier()
result = stressifier.apply_stress_marks("Привіт, як справи?")
print(result)
```