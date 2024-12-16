from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.utils.constants import TRAINED_MODELS_DIR, MODEL_NAME


class Generator:
    def __init__(self, filename: str = None):
        if filename:
            self.tokenizer = AutoTokenizer.from_pretrained(f"{TRAINED_MODELS_DIR}/{filename}/tokenizer/")
            model = AutoModelForCausalLM.from_pretrained(f"{TRAINED_MODELS_DIR}/{filename}/model/")
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
                r=32,
                lora_alpha=32,
                lora_dropout=0.1
            )
            self.model = get_peft_model(model, config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def generate(self, prompt: str) -> str:
        prompt = f"Generate a description for this product: {prompt}"

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=256,
            min_length=50,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return answer.replace(prompt, "").strip()
