---
layout: post
title: Preparing Translation Datasets
categories: [translation, data]
---


```python
# Install Hugging Face libraries
!pip install transformers datasets
```

    Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.37.2)
    Collecting datasets
      Downloading datasets-2.17.1-py3-none-any.whl (536 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m536.7/536.7 kB[0m [31m8.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.13.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.19.3 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.25.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (23.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2023.12.25)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.31.0)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.15.2)
    Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.2)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.2)
    Requirement already satisfied: pyarrow>=12.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (14.0.2)
    Requirement already satisfied: pyarrow-hotfix in /usr/local/lib/python3.10/dist-packages (from datasets) (0.6)
    Collecting dill<0.3.9,>=0.3.0 (from datasets)
      Downloading dill-0.3.8-py3-none-any.whl (116 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m13.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (1.5.3)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets) (3.4.1)
    Collecting multiprocess (from datasets)
      Downloading multiprocess-0.70.16-py310-none-any.whl (134 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m16.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: fsspec[http]<=2023.10.0,>=2023.1.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2023.6.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.9.3)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers) (4.9.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.2.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2023.4)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)
    Installing collected packages: dill, multiprocess, datasets
    Successfully installed datasets-2.17.1 dill-0.3.8 multiprocess-0.70.16

<!--more-->

```python
# Load dataset from the hub
from datasets import load_dataset

dataset = load_dataset("chittiman/BPCC_Telugu")
dataset['train'] = dataset['test']
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    Downloading readme:   0%|          | 0.00/28.0 [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/6.65G [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/512k [00:00<?, ?B/s]



    Downloading data:   0%|          | 0.00/1.53M [00:00<?, ?B/s]



    Generating train split: 0 examples [00:00, ? examples/s]



    Generating validation split: 0 examples [00:00, ? examples/s]



    Generating test split: 0 examples [00:00, ? examples/s]



```python
sample = dataset['train'][0]
sample
```




    {'eng': "Mom, let's go for a movie tomorrow.",
     'tel': 'అమ్మా, రేపు సినిమాకి వెళ్దాం.',
     'source': 'in22_conv'}




```python
# Convert dataset to OAI messages

def create_conversation(sample,src_lang='English',tgt_lang='Telugu'):
    src_sent = sample['eng']
    tgt_sent = sample['tel']
    return {"messages": [
                        {"role": "system", "content": "You are an AI translator who can translate text from English to Indic languages"},
                        {"role": "user", "content": f"Translate the following text from {src_lang} to {tgt_lang}.\n\n{src_lang}: {src_sent}\n{tgt_lang}: "},
                        {"role": "assistant", "content": tgt_sent}
                        ]
            }
sample_output = create_conversation(sample)
sample_output
```




    {'messages': [{'role': 'system',
       'content': 'You are an AI translator who can translate text from English to Indic languages'},
      {'role': 'user',
       'content': "Translate the following text from English to Telugu.\n\nEnglish: Mom, let's go for a movie tomorrow.\nTelugu: "},
      {'role': 'assistant', 'content': 'అమ్మా, రేపు సినిమాకి వెళ్దాం.'}]}




```python
print(sample_output['messages'][1]['content'] + sample_output['messages'][2]['content'])
```

    Translate the following text from English to Telugu.
    
    English: Mom, let's go for a movie tomorrow.
    Telugu: అమ్మా, రేపు సినిమాకి వెళ్దాం.



```python
dataset = dataset.map(create_conversation,batched=False).remove_columns(['eng','tel','source'])
```


```python
dataset['validation'][0]
```




    {'messages': [{'content': 'You are an AI translator who can translate text from English to Indic languages',
       'role': 'system'},
      {'content': 'Translate the following text from English to Telugu.\n\nEnglish: On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.\nTelugu: ',
       'role': 'user'},
      {'content': 'సోమవారం, స్టాన్ఫోర్డ్ యూనివర్శిటీ స్కూల్ ఆఫ్ మెడిసిన్ శాస్త్రవేత్తలు కణాల రకాన్ని క్రమబద్ధీకరించగల కొత్త రోగనిర్ధారణ సాధనం యొక్క ఆవిష్కరణను ప్రకటించారు: ప్రామాణిక ఇంక్జెట్ ప్రింటర్లను ఉపయోగించి 1 యు.ఎస్.',
       'role': 'assistant'}]}




```python
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
system_msg = 'You are an AI translator who can translate text from English to Indic Languages'
system_prompt = B_SYS + system_msg + E_SYS
print(system_prompt)
```

    <<SYS>>
    You are an AI translator who can translate text from English to Indic Languages
    <</SYS>>
    
    



```python
src_lang = 'English'
tgt_lang = 'Telugu'
context = f'Translate the following text from {src_lang} to {tgt_lang}\n'
print(context)
```

    Translate the following text from English to Telugu
    



```python
instruction = """
Context: {history} \n {context}
User: {question}
"""
```


```python
user_prompt = {}
```


```python
<s>[INST] <<SYS>>
System prompt
<</SYS>>

User prompt [/INST] Model answer </s>
```


```python
# Load dataset from the hub
dataset = load_dataset("b-mc2/sql-create-context", split="train")
dataset = dataset.shuffle().select(range(5))
dataset
```




    Dataset({
        features: ['answer', 'context', 'question'],
        num_rows: 5
    })




```python
sample = dataset[0]
sample
```




    {'answer': 'SELECT condition FROM table_1555308_1 WHERE bleeding_time = "Prolonged" AND prothrombin_time = "Unaffected"',
     'context': 'CREATE TABLE table_1555308_1 (condition VARCHAR, bleeding_time VARCHAR, prothrombin_time VARCHAR)',
     'question': 'In which condition(s) is bleeding time prolonged and prothrombin time unaffected?'}




```python

```


    Map:   0%|          | 0/5 [00:00<?, ? examples/s]





    {'messages': [{'content': 'You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_1555308_1 (condition VARCHAR, bleeding_time VARCHAR, prothrombin_time VARCHAR)',
       'role': 'system'},
      {'content': 'In which condition(s) is bleeding time prolonged and prothrombin time unaffected?',
       'role': 'user'},
      {'content': 'SELECT condition FROM table_1555308_1 WHERE bleeding_time = "Prolonged" AND prothrombin_time = "Unaffected"',
       'role': 'assistant'}]}




```python

```


```python
from datasets import load_dataset

# Convert dataset to OAI messages
system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""

def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message.format(schema=sample["context"])},
      {"role": "user", "content": sample["question"]},
      {"role": "assistant", "content": sample["answer"]}
    ]
  }

# Load dataset from the hub
dataset = load_dataset("b-mc2/sql-create-context", split="train")
dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
# split dataset into 10,000 training samples and 2,500 test samples
dataset = dataset.train_test_split(test_size=2500/12500)

print(dataset["train"][345]["messages"])

# save datasets to disk
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")
```


    Map:   0%|          | 0/12500 [00:00<?, ? examples/s]


    [{'content': 'You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_name_25 (no_1 VARCHAR, region__year_ VARCHAR)', 'role': 'system'}, {'content': 'What is No. 1, when Region (Year) is Mississippi (2010)?', 'role': 'user'}, {'content': 'SELECT no_1 FROM table_name_25 WHERE region__year_ = "mississippi (2010)"', 'role': 'assistant'}]



    Creating json from Arrow format:   0%|          | 0/10 [00:00<?, ?ba/s]



    Creating json from Arrow format:   0%|          | 0/3 [00:00<?, ?ba/s]





    1183830




```python
dataset['train'][0]
```




    {'messages': [{'content': 'You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_name_34 (pick INTEGER, college VARCHAR)',
       'role': 'system'},
      {'content': 'What is the pick number for New Mexico?', 'role': 'user'},
      {'content': 'SELECT AVG(pick) FROM table_name_34 WHERE college = "new mexico"',
       'role': 'assistant'}]}


