---
layout: post
title: "Building NMT for Languages the World Forgot: Achieving SOTA with Maithili and Konkani"
date: 2024-06-15
categories: [machine-learning, nlp, neural-machine-translation]
tags: [nmt, low-resource-languages, cross-lingual-transfer, back-translation, maithili, konkani]
---

## Introduction

Building effective neural translation models typically requires massive parallel datasets - a luxury most of the world's languages simply don't have. When the Indian Parliament initiated translation of proceedings into 22 regional languages, Maithili and Konkani presented the most fascinating challenge. With existing models barely scraping 0.3 BLEU scores, these extremely low-resource languages seemed nearly impossible to crack. By creating synthetic datasets and leveraging cross-lingual relationships, we pushed performance to 9.5+ BLEU - achieving state-of-the-art results with minimal resources. But why exactly were these languages so challenging to begin with?

## Problem Setup

To understand the scale of our challenge, consider the data landscape for English translation pairs: German has around 50 million sentence pairs, Hindi has roughly 5 million, while Tamil and Telugu have around 500,000 each. Assamese, already considered low-resource, has about 50,000 pairs. In comparison, Konkani had 5,000 parallel sentences available, and Maithili had 500.

Maithili, spoken in Bihar and Nepal, is linguistically related to Hindi, while Konkani, used in Goa and coastal areas, is connected to Marathi. These relationships offered potential for transfer learning. But the lack of native speakers internally, reliable benchmarks, and clean parallel data added challenges. However, all languages used Devanagari script, simplifying preprocessing. Before diving into our methods, it's worth understanding the encoder-decoder architecture that underlies modern NMT.

## Encoder-Decoder Primer

Neural machine translation relies on the encoder-decoder architecture, where an encoder module processes the input text in the source language and captures its semantic essence in vector form - essentially creating a numerical representation of the meaning. This encoded representation is then passed to the decoder module, which generates the translation in the target language through conditional next-token prediction, building the output one word at a time based on what it has already generated and the source encoding. Put simply, the encoder focuses on understanding meaning, while the decoder focuses on expressing that meaning in a new language.

The human parallel is striking: just as we can understand broken speech but struggle to speak fluently in a new language, encoders tolerate noise better than decoders. This creates an important asymmetry for low-resource NMT - source-side imperfections are manageable, but target-side noise directly degrades generation quality and makes pattern learning harder. Understanding this trade-off became central to our approach for Maithili and Konkani. Let's see how these insights shaped our methodology.

## Key Methods: Tagged Back-translation & Cross-lingual Transfer

Extreme data scarcity pushes us toward creative solutions in low-resource NMT. Two techniques that become essential are tagged back-translation and cross-lingual transfer. We'll examine these methods in general first to establish a clear foundation.

### Tagged Back-translation

Most parallel training data for translation comes from government documents, news articles, and published books - sources that inherently favor formal, literary language. This creates a fundamental mismatch with user expectations. Consider the difference:

Source: "The meeting will commence at 10 AM"
Formal: "बैठक सुबह 10 बजे आरंभ होगी" (using formal "आरंभ")  
Casual: "मीटिंग 10 बजे शुरू होगी" (using casual "मीटिंग" and "शुरू")

The challenge becomes: how do we create training data that reflects casual language use?

The solution lies in leveraging movie subtitles, which naturally contain casual, conversational language. We take existing Hindi subtitle data and reverse-translate it back to English using a trained Hindi-English model, creating synthetic English-Hindi pairs that preserve the casual tone. However, this synthetic data requires careful filtering to maintain quality. We apply several heuristics: removing exact duplicates, filtering out sentences that are too long, and eliminating pairs where the length ratio between source and target falls outside acceptable thresholds. This process yields a filtered dataset of casual-toned parallel data.

The next challenge is teaching the model to use this synthetic data effectively. Without guidance, the model cannot distinguish between original formal data and synthetic casual data during training, potentially learning confused representations that blend both styles. Tags solve this communication problem by explicitly signaling the data's nature and intended output style. We add `<bt>` tokens to warn the model that the source may be noisy due to back-translation artifacts - essentially telling it to be more tolerant of potential errors on the input side. Meanwhile, `<cas>` tokens indicate the target should use conversational language. During training, the model learns these associations: `<bt>` signals "expect possible noise in source, be forgiving" while `<cas>` signals "generate in casual style." This aligns with our earlier insight about encoder-decoder asymmetry - we can afford to be more tolerant on the source side while maintaining precision in generation. This allows us to control both noise handling and output tone at inference time.

Here's how this works in practice. During training, tagged data looks like this:

Training examples:
`<bt>` "I will reach there by 5 PM" → `<cas>` "मैं 5 बजे तक वहां पहुंच जाऊंगा"
`<bt>` "Please send me the documents today" → `<cas>` "आज मुझे डॉक्यूमेंट भेज देना"

At inference time, we control the output style by prefilling the first token:

Inference examples:
Input: "The train is delayed"
Output: "रेलगाड़ी में देरी हुई है" (formal, no prefill)

Input: "The train is delayed"
Directed output: `<cas>` "ट्रेन लेट हो गई है" (casual, with `<cas>` prefill)

The model learns to associate `<cas>` with natural, conversational Hindi while producing more formal translations when no style control is applied.

### Cross-lingual Transfer

Cross-lingual transfer leverages the linguistic relationships between related languages through joint training. Consider training a single model on both Hindi and Marathi data simultaneously. Despite being distinct languages, they share significant commonalities - overlapping vocabulary (पानी/पाणी for water), similar grammatical structures, and common root forms. During joint training, the model discovers these shared patterns and learns representations that benefit both languages, with the lower-resource language typically gaining more from this knowledge transfer. This works because related languages exist in overlapping semantic spaces, allowing the model to exploit commonalities for improved performance.

In practice, joint training uses language-specific target tags. Training data looks like this:

Training examples:
"I am going to school" → `<hi>` "मैं स्कूल जा रहा हूँ"
"I am going to school" → `<mr>` "मी शाळेत जात आहे"

At inference, we control the target language by prefilling with the appropriate tag:

Input: "The weather is nice today"
Directed output (Hindi): `<hi>` "आज मौसम अच्छा है"
Directed output (Marathi): `<mr>` "आज हवामान चांगले आहे"

## Adapting Techniques to Extreme Low-resource Settings

[Real-world implementation challenges and solutions:
- Specific difficulties you encountered with Maithili/Konkani
- How you adapted the general techniques
- Creative solutions you developed
- Decision-making process for your approach]

## Results & Insights

[Your SOTA achievement and key learnings:
- Quantitative results and improvements
- What worked better than expected
- Surprising findings
- Lessons learned from the process]

## Takeaways

[What practitioners can apply:
- Key principles for low-resource NMT
- When to use these techniques
- Practical advice for similar projects]

[Optional 1-2 sentences about social impact and language preservation]

---

*This work demonstrates that with creative application of existing techniques, we can build effective NMT systems even for extremely low-resource languages, opening doors for digital inclusion of underrepresented linguistic communities.*