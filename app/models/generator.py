from transformers import AutoTokenizer, AutoModelForCausalLM

from app.utils.constants import TRAINED_MODELS_DIR, MODEL_NAME


class Generator:
    def __init__(self, filename: str = None):
        if filename:
            self.tokenizer = AutoTokenizer.from_pretrained(f"{TRAINED_MODELS_DIR}/{filename}/tokenizer/")
            self.model = AutoModelForCausalLM.from_pretrained(f"{TRAINED_MODELS_DIR}/{filename}/model/")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def generate(self, prompt: str) -> str:
        prompt = f"Generate a description for this product: {prompt}"

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=256,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return answer.replace(prompt, "").strip()
