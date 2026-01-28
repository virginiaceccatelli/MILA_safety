# Multilingual Audio LLM Safety Project for COMP 400 - MILA

## Step 1. Translate JailbreakBench Prompts to German, Italian, English, Spanish and French 
Translate Gemma expected input, example: 


```
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": "cs",
                "target_lang_code": "de-DE",
                "text": text,
            }
        ],
    }
]
```


Translate Gemma expected pipeline, example: 

```
pipe = pipeline(
    "image-text-to-text",
    model="google/translategemma-27b-it",
    device="cuda",
    dtype=torch.bfloat16
)
```


## Step 2. Convert translations to audio 
-Qwen3-TTS: very heavy accents, relative refusal rate
- XTTS: much more native-sounding, relative refusal rate

## Step 3. Add code-switching to dataset, perturbations, etc. to examine their impact on safety 

