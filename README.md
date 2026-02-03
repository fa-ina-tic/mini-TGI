# mini-TGI
super lightweight inference server based on transformers

목표는 ... 최대한 많은 모델을 support 할 수 있는(transformers로 load하는 모든 모델을 목표로) 굉장히 가벼운 inference engine을 만드는 것.
그럼으로 (1) 특정 하드웨어에 종속된 최적화 방식(e.g, flash-attn2) 이나, 특정 모델 architecture 에 호환되는 최적화는 최대한 지양하겠습니다.

## Why we need it?
transformers 에 구현되어 있는 inference cli는 multi-gpu/multi-node를 지원 안함.
그리고 여러 양자화를 지원하냐...? 도 아님.
이거는 general 한 implementation으로 보기 어려움.

우리는 이 inference model로 하여금 최신 inference engine(e.g, vLLM, SGLang, TRT-LLM)에서 지원되지 않는 모델도 Serving 할 수 있는, PoC / benchmark에 적합한 inference engine 이 필요함.

현재 목표로 하는 최적화 내용은 아래와 같음.
- [ ] barebone with continuous batching & openai api compatible server
- [ ] multi-GPU / multi-Node inference using Accelerator
- [ ] Supports multiple quantization(e.g, GPTQ, AWQ)
- [ ] Supports multi modality
