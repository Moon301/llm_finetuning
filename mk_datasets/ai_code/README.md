
# PDF to Hugging Face Datasets Converter

PDF 문서를 Hugging Face datasets 포맷으로 변환하는 도구입니다. PDF에서 텍스트를 추출하고, 이를 instruction-tuning 형태의 데이터셋으로 변환합니다.

## 기능

- PDF 문서에서 텍스트 추출
- 텍스트를 청크로 분할
- Instruction-tuning 형태로 변환 (instruction, input, output)
- JSON/CSV 형식으로 저장
- 단일 파일 또는 디렉토리 일괄 처리
- OpenAI API를 통한 고급 instruction 생성 (선택사항)
- Ollama를 통한 로컬 모델 사용 (Qwen2.5:14b 등)

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 설정 (선택사항)

#### OpenAI API 사용
더 나은 instruction을 생성하려면:

```bash
# .env 파일 생성
cp env_example.txt .env

# .env 파일 편집하여 OpenAI API 키 설정
OPENAI_API_KEY=your_openai_api_key_here
```

#### Ollama 로컬 모델 사용
로컬에서 모델을 사용하려면:

```bash
# Ollama 설치 (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# 모델 다운로드
ollama pull qwen2.5:14b

# Ollama 서버 시작
ollama serve
```

## 사용법

### 1. 명령행 사용

#### 단일 PDF 파일 처리

```bash
python pdf_to_dataset.py --input sample.pdf --output dataset.json --format json
```

#### 디렉토리 내 모든 PDF 처리

```bash
python pdf_to_dataset.py --input pdfs/ --output output_datasets/ --format json
```

#### Ollama 로컬 모델 사용

```bash
python pdf_to_dataset.py --input sample.pdf --output dataset.json --ollama --ollama-model qwen2.5:14b
```

#### 옵션 설명

- `--input, -i`: 입력 PDF 파일 또는 디렉토리
- `--output, -o`: 출력 파일 경로
- `--format, -f`: 출력 형식 (json 또는 csv)
- `--api-key`: OpenAI API 키 (선택사항)
- `--max-chunks`: 최대 청크 수 (기본값: 50)
- `--ollama`: Ollama 모델 사용 (로컬)
- `--ollama-model`: Ollama 모델명 (기본값: qwen2.5:14b)

### 2. Python 코드 사용

```python
from pdf_to_dataset import PDFToDatasetConverter

# 변환기 초기화 (기본)
converter = PDFToDatasetConverter()

# OpenAI API 사용
converter = PDFToDatasetConverter(api_key="your_openai_api_key")

# Ollama 로컬 모델 사용
converter = PDFToDatasetConverter(use_ollama=True, ollama_model="qwen2.5:14b")

# 단일 PDF 처리
dataset = converter.create_dataset_from_pdf("sample.pdf", max_chunks=20)

# 저장
converter.save_dataset(dataset, "output.json", "json")
```

### 3. 예제 실행

```bash
python example_usage.py
```

## 출력 형식

생성되는 데이터셋은 다음과 같은 구조를 가집니다:

```json
{
  "instruction": "다음 텍스트의 주요 내용을 요약하세요.",
  "input": "고양이는 조용히 걸어갔다. 고양이는 매우 조용한 동물이다...",
  "output": "이 텍스트는 고양이의 특성에 대해 설명하고 있으며..."
}
```

## 주요 클래스

### PDFToDatasetConverter

PDF를 Hugging Face datasets 포맷으로 변환하는 메인 클래스입니다.

#### 주요 메서드

- `extract_text_from_pdf(pdf_path)`: PDF에서 텍스트 추출
- `split_text_into_chunks(text)`: 텍스트를 청크로 분할
- `generate_instruction_template(text_chunk)`: instruction 템플릿 생성
- `create_dataset_from_pdf(pdf_path, max_chunks)`: PDF를 데이터셋으로 변환
- `save_dataset(dataset, output_path, format)`: 데이터셋 저장
- `process_pdf_directory(input_dir, output_dir, format)`: 디렉토리 일괄 처리

## 설정 옵션

### 텍스트 분할 설정

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 청크 크기
    chunk_overlap=200,     # 청크 간 겹침
    length_function=len,   # 길이 측정 함수
)
```

### OpenAI API 사용

OpenAI API를 설정하면 더 나은 instruction을 생성할 수 있습니다:

```python
converter = PDFToDatasetConverter(api_key="your_api_key")
```

## 예제 데이터셋

### 기본 템플릿 (OpenAI API 없이)

```json
{
  "instruction": "다음 텍스트의 주요 내용을 요약하세요.",
  "input": "PDF에서 추출된 텍스트 청크...",
  "output": "이 텍스트는 X자로 구성되어 있으며, 주요 내용을 포함하고 있습니다."
}
```

### OpenAI API 사용 시

```json
{
  "instruction": "다음 문서에서 핵심 개념을 추출하고 정리하세요.",
  "input": "PDF에서 추출된 텍스트 청크...",
  "output": "이 문서에서 추출된 핵심 개념들: 1) 개념A, 2) 개념B, 3) 개념C..."
}
```

## 파일 구조

```
mk_datasets/
├── pdf_to_dataset.py      # 메인 변환 스크립트
├── example_usage.py       # 사용 예제
├── requirements.txt       # 의존성 패키지
├── env_example.txt       # 환경 변수 예제
└── README.md            # 이 파일
```

## 주의사항

1. **메모리 사용량**: 큰 PDF 파일을 처리할 때는 `max_chunks` 파라미터를 조정하세요.
2. **API 비용**: OpenAI API를 사용하면 비용이 발생할 수 있습니다.
3. **텍스트 품질**: PDF의 텍스트 품질에 따라 추출 결과가 달라질 수 있습니다.
4. **청크 크기**: 너무 작은 청크는 의미가 부족하고, 너무 큰 청크는 처리 시간이 오래 걸립니다.

## 문제 해결

### 일반적인 오류

1. **PDF 파일을 찾을 수 없음**
   - 파일 경로가 올바른지 확인
   - 파일이 실제로 존재하는지 확인

2. **OpenAI API 오류**
   - API 키가 올바르게 설정되었는지 확인
   - API 키의 잔액과 사용량 확인

3. **메모리 부족**
   - `max_chunks` 파라미터를 줄여보세요
   - 더 작은 PDF 파일로 테스트해보세요

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트나 기능 제안은 이슈를 통해 해주세요.

