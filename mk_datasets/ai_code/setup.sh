#!/bin/bash

echo "PDF to Hugging Face Datasets Converter 설치 및 설정"
echo "=================================================="

# 가상환경 생성 (선택사항)
read -p "가상환경을 생성하시겠습니까? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    python3 -m venv venv
    source venv/bin/activate
    echo "가상환경이 생성되고 활성화되었습니다."
fi

# 의존성 설치
echo "의존성 패키지 설치 중..."
pip install -r requirements.txt

# 환경 설정 파일 생성
if [ ! -f .env ]; then
    echo "환경 설정 파일 생성 중..."
    cp env_example.txt .env
    echo ".env 파일이 생성되었습니다. OpenAI API 키를 설정하려면 .env 파일을 편집하세요."
fi

echo ""
echo "설치가 완료되었습니다!"
echo ""
echo "사용법:"
echo "1. 단일 PDF 파일 처리:"
echo "   python pdf_to_dataset.py --input sample.pdf --output dataset.json"
echo ""
echo "2. 예제 실행:"
echo "   python example_usage.py"
echo ""
echo "3. 도움말 보기:"
echo "   python pdf_to_dataset.py --help" 