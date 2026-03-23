# 제목 생성 다시
def generate_title2(text, original_title):
    # 새로운 제목 생성 프롬프트
    title_prompt = (
      f"기존 본문: {text}\n\n"
      f"기존 제목: {original_title}\n\n"
      "당신은 기사 헤드라인 작성자입니다.\n"
      "기존 기사 제목, 기사 본문이 주어집니다.\n"
      "기사 본문을 기반으로 기존 기사 제목을 비선정적으로 다시 작성해주세요.\n"
      "선정적이지 않은 헤드라인으로 새롭게 생성해주어야 합니다.\n\n"
      "새로운 제목:\n"
    )


    # reason_prompt를 텐서로 변환하고, GPU로 이동
    title_inputs = tokenizer([title_prompt], return_tensors="pt", truncation=True)
    title_inputs = {key: value.to(device) for key, value in title_inputs.items()}

    # 선정적인 이유 생성
    title_ids = model.generate(
        title_inputs['input_ids'],
        # max_new_tokens=60,
        # num_beams=5,
        # repetition_penalty=2.5,
        # length_penalty=1.5,
        # no_repeat_ngram_size=1,
        # temperature=0.7,
        # early_stopping=True,
        # top_k=50,
        # top_p=0.9
        max_new_tokens=70,  # 생성할 최대 토큰 수
        early_stopping=True,  # 조기 종료 활성화
        do_sample=False,  # 가장 높은 확률의 토큰 선택하여 생성
        eos_token_id=2,  # 생성된 텍스트의 끝 토큰 ID
        # temperature=0.7
    )

    title = tokenizer.decode(title_ids[0], skip_special_tokens=True)

    return title
