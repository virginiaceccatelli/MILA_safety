# Multilingual Audio LLM Safety Project for COMP 400 - MILA

Audio language models are increasingly deployed in safety-critical settings, yet their safety alignment is
still primarily evaluated on monolingual harmful prompts. Their robustness to natural multilingual speech
remains limited and their stability under code-switched audio is also still poorly understood. Evaluating Large
Audio Language Models (LALMs) on multilingual code-switched speech is therefore important, as it represents
a natural and largely still unexplored jailbreak vector that can reveal safety weaknesses. With this study, we
construct and evaluate a multilingual, code-switched audio jailbreak dataset extended from JailbreakBench.
The prompts are translated into German, Spanish, Italian, and French, synthesised to speech with XTTS,
and evaluated on nine LALMs using an LLM-as-a-judge across Refusal Rate, Deflection Rate, and Jailbreak
Success Rate (JSR). We further introduce an augmented code-switched condition in which phonologically
plausible pseudo-words are inserted around safety-critical terms to test whether localized obfuscation increases
jailbreak success. Across models, code-switched harmful audio already yields substantially high JSR, with
non-English monolingual/non-English code-switched pairs producing the highest average attack success.
Pseudo-word insertion increases the JSR while reducing Refusal Rates, indicating that natural-sounding
obfuscation possibly weakens safety recognition. A separate study evaluates the same models using only
monolingual inputs on various comprehension datasets, namely MGSM (Multilingual Grade School Math),
Fleurs and SIB Fleurs-SLU, showing that the red-teaming failures cannot be explained solely by multilingual
audio misunderstanding, and that in some cases code-switched speech indeed constitutes a meaningful and
under-evaluated attack surface for audio model safety. Lastly, a simple prompt-based defense strategy is
proposed to strengthen the safety alignment on multilingual, code-switched and perturbated audio input.