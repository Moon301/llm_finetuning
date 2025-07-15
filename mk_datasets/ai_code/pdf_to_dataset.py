import os
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import logging
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from datasets import Dataset
import openai
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class PDFToDatasetConverter:
    def __init__(self, api_key: str = None, use_ollama: bool = False, ollama_model: str = "qwen2.5:14b"):
        """
        PDF를 Hugging Face datasets 포맷으로 변환하는 클래스
        
        Args:
            api_key: OpenAI API 키 (환경변수에서 자동 로드됨)
            use_ollama: Ollama 사용 여부
            ollama_model: 사용할 Ollama 모델명
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        
        if self.api_key and not self.use_ollama:
            openai.api_key = self.api_key
            logger.info("OpenAI API를 사용합니다.")
        elif self.use_ollama:
            logger.info(f"Ollama 모델을 사용합니다: {self.ollama_model}")
        else:
            logger.warning("API 키가 설정되지 않았습니다. 기본 템플릿을 사용합니다.")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        PDF 파일에서 텍스트를 추출합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # 모든 페이지의 텍스트를 결합
            full_text = ""
            for page in pages:
                full_text += page.page_content + "\n"
            
            logger.info(f"PDF에서 {len(full_text)} 문자를 추출했습니다.")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 중 오류 발생: {e}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[Document]:
        """
        텍스트를 청크로 분할합니다.
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 문서 청크들
        """
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        logger.info(f"텍스트를 {len(documents)}개의 청크로 분할했습니다.")
        return documents
    
    def generate_instruction_template(self, text_chunk: str) -> Dict[str, str]:
        """
        텍스트 청크를 기반으로 instruction 템플릿을 생성합니다.
        
        Args:
            text_chunk: 텍스트 청크
            
        Returns:
            instruction, input, output을 포함한 딕셔너리
        """
        if not self.api_key and not self.use_ollama:
            # 기본 템플릿 사용 - PDF 내용에 대한 질문-답변 형태
            return self._generate_basic_qa_template(text_chunk)
        
        try:
            if self.use_ollama:
                # Ollama용 프롬프트
                prompt = f"""다음 텍스트를 기반으로 사용자가 물어볼 수 있는 질문과 그에 대한 답변을 생성하세요.

텍스트:
{text_chunk[:1000]}

다음 형식으로 응답해주세요:
instruction: 다음 문서의 내용에 대해 질문하세요.
input: [실용적인 질문]
output: [텍스트 내용을 바탕으로 한 답변]

질문은 텍스트의 내용을 이해하고 활용할 수 있는 실용적인 질문이어야 합니다."""
                response_text = self._call_ollama(prompt)
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "주어진 텍스트를 기반으로 사용자가 물어볼 수 있는 질문과 그에 대한 답변을 생성하세요. 질문은 텍스트의 내용을 이해하고 활용할 수 있는 실용적인 질문이어야 합니다."
                        },
                        {
                            "role": "user",
                            "content": f"다음 텍스트를 기반으로 instruction(질문), input(질문), output(답변)을 생성하세요:\n\n{text_chunk[:1000]}"
                        }
                    ],
                    max_tokens=400,
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
            
            # 응답을 파싱하여 구조화된 데이터로 변환
            lines = response_text.split('\n')
            instruction = ""
            input_text = ""
            output = ""
            
            for line in lines:
                if line.startswith("instruction:"):
                    instruction = line.replace("instruction:", "").strip()
                elif line.startswith("input:"):
                    input_text = line.replace("input:", "").strip()
                elif line.startswith("output:"):
                    output = line.replace("output:", "").strip()
            
            if not instruction:
                instruction = "다음 문서의 내용에 대해 질문하세요."
            if not input_text:
                input_text = self._generate_question_from_text(text_chunk)
            if not output:
                output = self._generate_answer_from_text(text_chunk, input_text)
            
            return {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
            
        except Exception as e:
            logger.error(f"API 호출 중 오류 발생: {e}")
            # 기본 템플릿 사용
            return self._generate_basic_qa_template(text_chunk)
    
    def _generate_basic_qa_template(self, text_chunk: str) -> Dict[str, str]:
        """
        기본 질문-답변 템플릿을 생성합니다.
        """
        # 텍스트에서 키워드 추출
        keywords = self._extract_keywords(text_chunk)
        
        # 질문 생성
        if "계약" in text_chunk or "계약서" in text_chunk:
            question = "이 문서에서 계약 관련 주요 내용은 무엇인가요?"
        elif "입주" in text_chunk or "주택" in text_chunk:
            question = "이 문서에서 입주 관련 조건이나 절차는 어떻게 되나요?"
        elif "신청" in text_chunk or "지원" in text_chunk:
            question = "이 문서에서 신청이나 지원 관련 내용은 무엇인가요?"
        elif "조건" in text_chunk or "요건" in text_chunk:
            question = "이 문서에서 주요 조건이나 요건은 무엇인가요?"
        elif "기간" in text_chunk or "일정" in text_chunk:
            question = "이 문서에서 주요 기간이나 일정은 어떻게 되나요?"
        else:
            question = f"이 문서에서 {keywords[0] if keywords else '주요'} 관련 내용은 무엇인가요?"
        
        # 답변 생성
        answer = self._generate_simple_answer(text_chunk, question)
        
        return {
            "instruction": "다음 문서의 내용에 대해 질문하세요.",
            "input": question,
            "output": answer
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        텍스트에서 주요 키워드를 추출합니다.
        """
        # 간단한 키워드 추출 (실제로는 더 정교한 방법 사용 가능)
        keywords = []
        important_words = ["계약", "입주", "신청", "조건", "기간", "주택", "지원", "절차", "요건"]
        
        for word in important_words:
            if word in text:
                keywords.append(word)
        
        return keywords[:3]  # 최대 3개 키워드 반환
    
    def _generate_question_from_text(self, text_chunk: str) -> str:
        """
        텍스트를 기반으로 질문을 생성합니다.
        """
        if len(text_chunk) < 50:
            return "이 문서의 주요 내용은 무엇인가요?"
        
        # 텍스트의 첫 부분을 사용하여 질문 생성
        first_sentence = text_chunk.split('.')[0] + '.'
        
        if "계약" in first_sentence:
            return "이 문서의 계약 관련 주요 내용을 설명해주세요."
        elif "입주" in first_sentence:
            return "이 문서에서 입주 관련 조건이나 절차는 어떻게 되나요?"
        elif "신청" in first_sentence:
            return "이 문서에서 신청이나 지원 관련 내용은 무엇인가요?"
        else:
            return "이 문서의 주요 내용을 요약해주세요."
    
    def _generate_answer_from_text(self, text_chunk: str, question: str) -> str:
        """
        텍스트와 질문을 기반으로 답변을 생성합니다.
        """
        # 질문 유형에 따라 답변 생성
        if "계약" in question:
            return f"이 문서에 따르면, {text_chunk[:200]}... (계약 관련 내용)"
        elif "입주" in question:
            return f"입주 관련 내용: {text_chunk[:200]}... (입주 조건 및 절차)"
        elif "신청" in question or "지원" in question:
            return f"신청/지원 관련 내용: {text_chunk[:200]}... (신청 절차 및 지원 내용)"
        elif "조건" in question or "요건" in question:
            return f"주요 조건/요건: {text_chunk[:200]}... (필요 조건들)"
        elif "기간" in question or "일정" in question:
            return f"주요 기간/일정: {text_chunk[:200]}... (관련 일정들)"
        else:
            return f"이 문서의 주요 내용: {text_chunk[:300]}..."
    
    def _generate_simple_answer(self, text_chunk: str, question: str) -> str:
        """
        간단한 답변을 생성합니다.
        """
        # 텍스트 길이에 따라 답변 조정
        if len(text_chunk) < 100:
            return f"이 문서는 {text_chunk}에 대한 내용을 담고 있습니다."
        
        # 질문 유형에 따른 답변
        if "계약" in question:
            return f"계약 관련 내용: {text_chunk[:200]}..."
        elif "입주" in question:
            return f"입주 관련 내용: {text_chunk[:200]}..."
        elif "신청" in question:
            return f"신청 관련 내용: {text_chunk[:200]}..."
        elif "조건" in question or "요건" in question:
            return f"주요 조건/요건: {text_chunk[:200]}..."
        else:
            return f"이 문서의 주요 내용: {text_chunk[:250]}..."
    
    def create_dataset_from_pdf(self, pdf_path: str, max_chunks: int = 50) -> Dataset:
        """
        PDF 파일을 Hugging Face datasets 포맷으로 변환합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            max_chunks: 최대 청크 수 (메모리 관리를 위해)
            
        Returns:
            Hugging Face Dataset 객체
        """
        # PDF에서 텍스트 추출
        text = self.extract_text_from_pdf(pdf_path)
        
        # 텍스트를 청크로 분할
        documents = self.split_text_into_chunks(text)
        
        # 최대 청크 수 제한
        documents = documents[:max_chunks]
        
        # 각 청크를 instruction 형태로 변환
        dataset_items = []
        for i, doc in enumerate(documents):
            logger.info(f"청크 {i+1}/{len(documents)} 처리 중...")
            
            item = self.generate_instruction_template(doc.page_content)
            dataset_items.append(item)
        
        # DataFrame으로 변환
        df = pd.DataFrame(dataset_items)
        
        # Hugging Face Dataset으로 변환
        dataset = Dataset.from_pandas(df)
        
        logger.info(f"데이터셋 생성 완료: {len(dataset)} 개의 항목")
        return dataset
    
    def save_dataset(self, dataset: Dataset, output_path: str, format: str = "json"):
        """
        데이터셋을 파일로 저장합니다.
        
        Args:
            dataset: 저장할 데이터셋
            output_path: 출력 파일 경로
            format: 저장 형식 ("json" 또는 "csv")
        """
        if format.lower() == "json":
            # pandas DataFrame으로 변환 후 저장 (한글 깨짐 방지)
            df = dataset.to_pandas()
            df.to_json(output_path, force_ascii=False, orient="records", lines=True)
            logger.info(f"데이터셋을 JSON 형식(한글)으로 저장했습니다: {output_path}")
        elif format.lower() == "csv":
            # CSV 형식으로 저장
            dataset.to_csv(output_path)
            logger.info(f"데이터셋을 CSV 형식으로 저장했습니다: {output_path}")
        else:
            raise ValueError("지원되지 않는 형식입니다. 'json' 또는 'csv'를 사용하세요.")
    
    def process_pdf_directory(self, input_dir: str, output_dir: str, format: str = "json"):
        """
        디렉토리 내의 모든 PDF 파일을 처리합니다.
        
        Args:
            input_dir: PDF 파일들이 있는 입력 디렉토리
            output_dir: 결과를 저장할 출력 디렉토리
            format: 저장 형식
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"{input_dir}에서 PDF 파일을 찾을 수 없습니다.")
            return
        
        for pdf_file in pdf_files:
            logger.info(f"처리 중: {pdf_file}")
            
            try:
                # 데이터셋 생성
                dataset = self.create_dataset_from_pdf(str(pdf_file))
                
                # 출력 파일명 생성
                output_filename = pdf_file.stem + f".{format}"
                output_file = output_path / output_filename
                
                # 저장
                self.save_dataset(dataset, str(output_file), format)
                
            except Exception as e:
                logger.error(f"{pdf_file} 처리 중 오류 발생: {e}")

    def _call_ollama(self, prompt: str) -> str:
        """
        Ollama API를 호출하여 응답을 받습니다.
        
        Args:
            prompt: 전송할 프롬프트
            
        Returns:
            모델의 응답
        """
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 400
                }
            }
            
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 호출 중 오류 발생: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama 응답 처리 중 오류 발생: {e}")
            raise


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF를 Hugging Face datasets 포맷으로 변환")
    parser.add_argument("--input", "-i", required=True, help="입력 PDF 파일 또는 디렉토리")
    parser.add_argument("--output", "-o", required=True, help="출력 파일 경로")
    parser.add_argument("--format", "-f", default="json", choices=["json", "csv"], 
                       help="출력 형식 (json 또는 csv)")
    parser.add_argument("--api-key", help="OpenAI API 키 (선택사항)")
    parser.add_argument("--max-chunks", type=int, default=50, 
                       help="최대 청크 수 (기본값: 50)")
    parser.add_argument("--ollama", action="store_true", help="Ollama 모델을 사용하여 instruction을 생성합니다.")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:14b", help="Ollama 모델명 (기본값: qwen2.5:14b)")
    
    args = parser.parse_args()
    
    # 변환기 초기화
    converter = PDFToDatasetConverter(api_key=args.api_key, use_ollama=args.ollama, ollama_model=args.ollama_model)
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # 단일 PDF 파일 처리
        logger.info(f"단일 PDF 파일 처리: {input_path}")
        dataset = converter.create_dataset_from_pdf(str(input_path), args.max_chunks)
        converter.save_dataset(dataset, args.output, args.format)
        
    elif input_path.is_dir():
        # 디렉토리 내 모든 PDF 처리
        logger.info(f"디렉토리 내 PDF 파일들 처리: {input_path}")
        converter.process_pdf_directory(str(input_path), args.output, args.format)
        
    else:
        logger.error("유효하지 않은 입력 경로입니다. PDF 파일 또는 디렉토리를 지정하세요.")


if __name__ == "__main__":
    main() 