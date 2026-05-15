# LLM Project
Python과 PyTorch를 사용해 LLM을 기초부터 직접 구현하며 학습하는 프로젝트입니다.

## 목표
이 프로젝트의 목표는 완성된 라이브러리를 바로 사용하는것이 아니라, 언어 모델이 동작하는 과정을 작은 단위부터 직접 구현하며 이해하는 것입니다.

주요 목표는 다음과 같습니다.
- PyTorch 기본 사용법을 코드로 익히기
- 텍스트를 토큰으로 변환하는 과정 이해하시
- 다음 토큰 예측 방식으로 언어 모델 학습하기
- Bigram 모델부터 시작해 Transformer 구조까지 확장하기
- 최종적으로 작은 GPT-style casual language model 구현하기

## 개발 환경
- Python 3.14
- NumPy
- PyTorch
- Apple Silicon MPS 사용 가능시 GPU 가속

## 환경 설정
가상환경을 생성합니다.
```bash
python3.14 -m venv .venv
```

가상환경을 활성화합니다.
```bash
source .venv/bin/activate
```

프로젝트 의존성을 설치합니다.
```bash
pip install -e ".[dev]"
```

설치 확인 예시입니다.
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.backends.mps.is_available())"
```

## 프로젝트 구조
```text
  LLM-Project/
  ├── CODEX_NOTES.md
  ├── README.md
  ├── pyproject.toml
  ├── data/
  │   ├── raw/
  │   └── processed/
  ├── scripts/
  ├── src/
  │   └── llm_project/
  └── tests/
```

## 학습 순서

1. PyTorch Tensor 기초
2. Character-level tokenizer
3. Dataset과 batch 생성
4. Bigram language model
5. 학습 루프
6. 텍스트 생성
7. Self-Attention
8. Multi-Head Attention
9. Transformer Block
10. GPT-style causal language model

## 현재 단계
현재는 프로젝트 기본 구조와 Python 실행 환경을 준비하는단계입니다.
첫 구현은 character-level tokenizer부터 시작합니다.