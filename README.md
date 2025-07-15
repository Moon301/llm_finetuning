## llm_finetuning

**llm 파인튜닝 프로젝트**
특정 도메인의 지식을 학습한 llm 모델 구축(ex 반도체 데이터, 전기차 데이터 전문 등)
시계열 데이터의 분석 및 예측을 학습한 전용 모델 구축

**목적**
산업 데이터의 검색, 분석을 위해서 RAG 기법 외의 파인튜닝을 통한 모델 성능을 개선해보는 것이 목적


**목표**
1. llm이 학습 가능한 데이터셋 구축(과제 질의응답 ~ 산업 데이터)
2. 학습 성능 검증(기본모델+RAG와 학습모델+RAG의 성능 차이)
3. 1차목표 과제에 대한 Q&A 모델 개발 &rarr; RAG없이 기본 학습 능력으로만

**Think**
- 일반적인 llm 파인튜닝 학습 방법 확인
- llm모델의 시계열 데이터 학습 방법?


### Unsloth 라이브러리 활용 LLM finetuning
- [테스트 코드](https://github.com/Moon301/llm_finetuning/blob/main/unsloth/ft_llama_test.ipynb)
    - Alpaca Style 형식의 데이터 셋을 구축하고 단일 GPU로 학습 진행
    - HuggingFace에 FT된 모델 업로드 [moon301/llama-3.1-8b-finetuned-lh-announcement](https://huggingface.co/moon301/llama-3.1-8b-finetuned-lh-announcement)
    - 데이터셋의 질과 양에 따라서 모델의 학습 능력이 달라질 것 같음
    - 해당 모델은 Llama3 8B모델을 4bit로 양자화된 모델(unsloth 제공) + LoRA기법 활용

### 공통 SW 개발 추진 주요 (목표) 기술

(1) 산업 데이터 전처리
(2) 데이터 분석 및 특징 확인
(3) 데이터 특징별 학습, 판단 자동화

(4) LLM 기반 데이터 분석
(5) LLM / 벡터DB 연계

(6) LLM 기반 Q&A 시스템
(7) Q&A 기반 사용자 작업 지원 SW

(8) LLM 기반 산업 데이터 검색, 분석, 트래킹
(9) LLM 기반 에이전트 (데이터 검색, 필터링, 자율분석)


