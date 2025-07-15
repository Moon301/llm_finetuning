#!/usr/bin/env python3
"""
PDF를 Hugging Face datasets 포맷으로 변환하는 사용 예제
"""

import os
from pathlib import Path
from pdf_to_dataset import PDFToDatasetConverter

def example_single_pdf():
    """단일 PDF 파일 처리 예제"""
    print("=== 단일 PDF 파일 처리 예제 ===")
    
    # 변환기 초기화 (OpenAI API 키가 설정되어 있다면 자동으로 사용됨)
    converter = PDFToDatasetConverter()
    
    # PDF 파일 경로 (실제 PDF 파일 경로로 변경하세요)
    pdf_path = "/home/moon/llm_finetuning/mk_datasets/sample/입주자모집공고.pdf"  # 실제 PDF 파일 경로로 변경
    
    if not os.path.exists(pdf_path):
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        print("샘플 PDF 파일을 준비한 후 다시 실행하세요.")
        return
    
    try:
        # 데이터셋 생성
        dataset = converter.create_dataset_from_pdf(pdf_path, max_chunks=20)
        
        # JSON 형식으로 저장
        converter.save_dataset(dataset, "output_dataset.json", "json")
        
        # CSV 형식으로도 저장
        converter.save_dataset(dataset, "output_dataset.csv", "csv")
        
        print(f"데이터셋 생성 완료!")
        print(f"- JSON 파일: output_dataset.json")
        print(f"- CSV 파일: output_dataset.csv")
        print(f"- 총 항목 수: {len(dataset)}")
        
        # 샘플 데이터 출력
        print("\n=== 샘플 데이터 ===")
        for i, item in enumerate(dataset):
            if i >= 3:  # 처음 3개만 출력
                break
            print(f"항목 {i+1}:")
            print(f"  Instruction: {item['instruction']}")
            print(f"  Input: {item['input'][:100]}...")
            print(f"  Output: {item['output'][:100]}...")
            print()
            
    except Exception as e:
        print(f"오류 발생: {e}")

def example_directory_processing():
    """디렉토리 내 모든 PDF 처리 예제"""
    print("=== 디렉토리 내 PDF 파일들 처리 예제 ===")
    
    converter = PDFToDatasetConverter()
    
    # 입력 디렉토리 (PDF 파일들이 있는 디렉토리)
    input_dir = "pdfs"  # 실제 PDF 파일들이 있는 디렉토리 경로로 변경
    output_dir = "output_datasets"
    
    if not os.path.exists(input_dir):
        print(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        print("PDF 파일들이 있는 디렉토리를 준비한 후 다시 실행하세요.")
        return
    
    try:
        # 디렉토리 내 모든 PDF 처리
        converter.process_pdf_directory(input_dir, output_dir, "json")
        
        print(f"디렉토리 처리 완료!")
        print(f"- 입력 디렉토리: {input_dir}")
        print(f"- 출력 디렉토리: {output_dir}")
        
        # 생성된 파일들 확인
        output_path = Path(output_dir)
        if output_path.exists():
            files = list(output_path.glob("*.json"))
            print(f"- 생성된 파일 수: {len(files)}")
            for file in files:
                print(f"  - {file.name}")
                
    except Exception as e:
        print(f"오류 발생: {e}")

def example_with_openai():
    """OpenAI API를 사용한 고급 처리 예제"""
    print("=== OpenAI API를 사용한 고급 처리 예제 ===")
    
    # OpenAI API 키 설정 (환경변수 또는 직접 설정)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("OpenAI API 키가 설정되지 않았습니다.")
        print("환경변수 OPENAI_API_KEY를 설정하거나 .env 파일에 추가하세요.")
        return
    
    converter = PDFToDatasetConverter(api_key=api_key)
    
    # PDF 파일 경로
    pdf_path = "sample.pdf"  # 실제 PDF 파일 경로로 변경
    
    if not os.path.exists(pdf_path):
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    try:
        # 더 많은 청크로 처리 (더 풍부한 데이터셋 생성)
        dataset = converter.create_dataset_from_pdf(pdf_path, max_chunks=100)
        
        # 저장
        converter.save_dataset(dataset, "advanced_dataset.json", "json")
        
        print(f"고급 데이터셋 생성 완료!")
        print(f"- 파일: advanced_dataset.json")
        print(f"- 총 항목 수: {len(dataset)}")
        
    except Exception as e:
        print(f"오류 발생: {e}")

def example_with_ollama():
    """Ollama를 사용한 로컬 모델 처리 예제"""
    print("=== Ollama를 사용한 로컬 모델 처리 예제 ===")
    
    # Ollama 사용 여부 확인
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            print(f"사용 가능한 Ollama 모델: {available_models}")
            
            if "qwen2.5:14b" not in available_models:
                print("qwen2.5:14b 모델이 설치되지 않았습니다.")
                print("다음 명령어로 모델을 설치하세요:")
                print("ollama pull qwen2.5:14b")
                return
        else:
            print("Ollama 서버에 연결할 수 없습니다.")
            print("Ollama가 실행 중인지 확인하세요.")
            return
    except Exception as e:
        print(f"Ollama 연결 확인 중 오류: {e}")
        print("Ollama가 설치되어 있고 실행 중인지 확인하세요.")
        return
    
    # Ollama를 사용하는 변환기 초기화
    converter = PDFToDatasetConverter(use_ollama=True, ollama_model="qwen2.5:14b")
    
    # PDF 파일 경로
    pdf_path = "sample.pdf"  # 실제 PDF 파일 경로로 변경
    
    if not os.path.exists(pdf_path):
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    try:
        # 데이터셋 생성
        dataset = converter.create_dataset_from_pdf(pdf_path, max_chunks=10)
        
        # 저장
        converter.save_dataset(dataset, "ollama_dataset.json", "json")
        
        print(f"Ollama 모델을 사용한 데이터셋 생성 완료!")
        print(f"- 파일: ollama_dataset.json")
        print(f"- 총 항목 수: {len(dataset)}")
        
        # 샘플 데이터 출력
        print("\n=== Ollama 생성 샘플 데이터 ===")
        for i, item in enumerate(dataset):
            if i >= 2:  # 처음 2개만 출력
                break
            print(f"항목 {i+1}:")
            print(f"  Instruction: {item['instruction']}")
            print(f"  Input: {item['input']}")
            print(f"  Output: {item['output'][:150]}...")
            print()
        
    except Exception as e:
        print(f"오류 발생: {e}")

def create_sample_pdf():
    """테스트용 샘플 PDF 생성 (선택사항)"""
    print("=== 테스트용 샘플 PDF 생성 ===")
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # 간단한 샘플 PDF 생성
        c = canvas.Canvas("sample.pdf", pagesize=letter)
        
        # 제목
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "샘플 문서")
        
        # 본문
        c.setFont("Helvetica", 12)
        text_content = [
            "이것은 테스트용 샘플 문서입니다.",
            "이 문서는 PDF를 데이터셋으로 변환하는 기능을 테스트하기 위해 생성되었습니다.",
            "",
            "주요 내용:",
            "1. PDF 텍스트 추출",
            "2. 텍스트 청크 분할",
            "3. Instruction 형태로 변환",
            "4. Hugging Face datasets 포맷으로 저장",
            "",
            "이 과정을 통해 PDF 문서를 머신러닝 모델 학습에 사용할 수 있는",
            "구조화된 데이터셋으로 변환할 수 있습니다."
        ]
        
        y_position = 700
        for line in text_content:
            c.drawString(100, y_position, line)
            y_position -= 20
        
        c.save()
        print("샘플 PDF 파일이 생성되었습니다: sample.pdf")
        
    except ImportError:
        print("reportlab 패키지가 설치되지 않았습니다.")
        print("pip install reportlab로 설치하거나, 기존 PDF 파일을 사용하세요.")
    except Exception as e:
        print(f"샘플 PDF 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    print("PDF를 Hugging Face datasets 포맷으로 변환하는 예제")
    print("=" * 50)
    
    # 샘플 PDF 생성 (선택사항)
    create_sample_pdf()
    
    print("\n" + "=" * 50)
    
    # 예제 실행
    example_single_pdf()
    
    print("\n" + "=" * 50)
    
    # 디렉토리 처리 예제 (PDF 파일들이 있는 디렉토리가 있다면)
    if os.path.exists("pdfs"):
        example_directory_processing()
    else:
        print("'pdfs' 디렉토리가 없어서 디렉토리 처리 예제를 건너뜁니다.")
    
    print("\n" + "=" * 50)
    
    # OpenAI API 예제
    if os.getenv("OPENAI_API_KEY"):
        example_with_openai()
    else:
        print("OpenAI API 키가 설정되지 않아서 고급 처리 예제를 건너뜁니다.")
    
    print("\n" + "=" * 50)
    
    # Ollama API 예제
    example_with_ollama()
    
    print("\n모든 예제 실행 완료!") 