## 데이터 셋 구축 가이드

데이터 셋이란 LLM에게 학습하는데 사용할 수 있는 데이터 모음.
학습에 유용하려면 텍스트 데이터가 토큰화 가능한 형식이어ㅓ야 한다.


Toknization &rarr; 텍스트를 단어, 서브워드, 문자 단위의 토큰으로 분리하여 LLM이 텍스트를 효과적으로 처리할 수 있도록 도와줌


**Data Format**

| **형식 (Format)** | **설명 (Description)** | **훈련 방식 (Training Type)** |
| --- | --- | --- |
| **Raw Corpus** | 웹사이트, 책, 기사 등에서 가져온 **원시 텍스트(raw text)**. | **Continued Pretraining (CPT)**지속 사전학습 |
| **Instruct** | 모델이 따라야 할 **명령어(instruction)**와, 목표로 하는 **출력 예시(output example)**. | **Supervised Fine-Tuning (SFT)**지도 미세조정 |
| **Conversation** | 사용자와 AI 어시스턴트 간의 **다중 턴 대화**. | **Supervised Fine-Tuning (SFT)** |
| **RLHF** | 사용자와 AI 어시스턴트 간의 대화이며, **AI의 응답을 평가자(스크립트/모델/사람)가 순위화**함. | **Reinforcement Learning (RL)**강화학습 |


**데이터 셋의 목적**

- 어떤 데이터가 필요한지, 어떤 포맷을 사용할지 결정하는데 도움이 된다.
- 요약같은 새로운 작업에 모델을 적응시키거나, 특정 캐릭터를 연기 하는 능력을 향상 시키는 것이 될 수 있다.
- 채팅 기반 대화: 질문-응답, 외국어 학습, 고객 지원, 일반적인 대화 등
- 구조화된 작업: 분류, 요약, 생성 작업 등
- 도메인 특화 데이터: 의료, 금융, 기술 분야 등의 특화된 주제

→ 어떤 목적인지에 따라 필요한 데이터의 내용이 달라지고, 사용할 데이터의 형식도 달라진다.

출력형식

- 우리가 목표한 결과를 달성하기 위한 출력형식
- ex) json, html, text, code 또는 영어, 한국어, 일어
- 출력 스타일은 단순한 포맷 뿐만아니라 언어 또는 콘첸츠 측성(코드/에세이/표)까지 포함

→ 원하는 출력이 무엇인지에 따라 사용해야할 데이터 소스가 달라지고, 전처리 방식이나 토큰화 방식도  달라질 수 있다.

데이터 품질

- 데이터의 품질과 양을 분석해야 한다.
- 오픈소스 데이터: Hugging Face Datasets, WikiPedia(언어모델 학습)
- 데이터셋을 만드는 가장 효과적인 방법 중 하나는 HuggingFace의 Share GPT와 같은 범용 데이터셋과 결합하는 것


**데이터 학습 포멧의 종류**
1. Continued Pretraining (지속 사전학습)

- **목적**: 기존의 사전학습된 LLM에 특정 도메인(예: 법률, 의학 등)의 텍스트를 추가로 학습시켜 **언어적 적응력을 높이는 것**
- ex :   "text": "Pasta carbonara is a traditional Roman pasta dish. The sauce is made by mixing raw eggs with grated Pecorino Romano cheese and black pepper. The hot pasta is then tossed with crispy guanciale (cured pork cheek) and the egg mixture, creating a creamy sauce from the residual heat. Despite popular belief, authentic carbonara never contains cream or garlic. The dish likely originated in Rome in the mid-20th century, though its exact origins are debated...”
- 자연어 흐름을 그대로 유지하기 때문에 모델이 연속적인 문맥과 언어 패턴을 효과적으로 학습
- 모델을 **특정 작업(task)에 적응**시키고자 하고, 지정된 지시(instruction)에 따라 단일 응답(single-turn output)을 생성하도록 학습시키려는 경우에는 **Alpaca 스타일의 Instruction 포맷**을 사용할 수 있습니다.


1. `Alpaca Style` - instruction, input, output
    
    ```
    "Instruction": "Task we want the model to perform."
    
    "Input": "Optional, but useful, it will essentially be the user's query."
    
    "Output": "The expected result of the task and the output of the model."
    ```


2. `Share GPT`  - 템플릿 포맷(from / value)
    
    ```
    {
      "conversations": [
        {
          "from": "human",
          "value": "Can you help me make pasta carbonara?"
        },
        {
          "from": "gpt",
          "value": "Would you like the traditional Roman recipe, or a simpler version?"
        },
        {
          "from": "human",
          "value": "The traditional version please"
        },
        {
          "from": "gpt",
          "value": "The authentic Roman carbonara uses just a few ingredients: pasta, guanciale, eggs, Pecorino Romano, and black pepper. Would you like the detailed recipe?"
        }
      ]
    }
    ```


3. `OpenAI ChatML` - hugging Face에서 기본 값으로 사용되는 형식 (role, content)
    
    ```
    {
      "messages": [
        {
          "role": "user",
          "content": "What is 1+1?"
        },
        {
          "role": "assistant",
          "content": "It's 2!"
        },
      ]
    }
    ```

