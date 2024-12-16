import os
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk, Dataset
import wandb

from app.utils.constants import (
    TRAINED_MODELS_DIR, MODEL_NAME,
    DATASET_PATH, FINE_TUNING_OUTPUT_DIR,
    BATCH_SIZE, NUM_TRAIN_EPOCHS, LOG_DIR,
)

def clean_file(filename):
    try:
        clean_file_path = f"{DATASET_PATH}/{os.path.splitext(filename)[0]}.cleaned.csv"

        if os.path.exists(clean_file_path):
            print(f"Limpando arquivo existente: {clean_file_path}")
            os.remove(clean_file_path)

        print("Lendo o arquivo JSON...")
        data = pd.read_json(f"{DATASET_PATH}/{filename}", lines=True)

        print("Removendo duplicatas com base nas colunas 'title' e 'content'...")
        data = data.drop_duplicates(subset=["title", "content"])

        print("Filtrando 'title' com pelo menos 4 caracteres...")
        data = data[data["title"].str.len() > 3]

        print("Filtrando 'content' com pelo menos 4 caracteres...")
        data = data[data["content"].str.len() > 3]

        print("Renomeando colunas para 'prompt' e 'answer'...")
        data = data.rename(columns={"title": "prompt", "content": "answer"})

        print("Removendo valores NaN nas colunas 'prompt' e 'answer'...")
        data = data.replace("nan", pd.NA)
        data = data.dropna(how="all")

        print("Removendo valores nulos nas colunas 'prompt' e 'answer'...")
        data = data[~data["prompt"].isnull() & ~data["answer"].isnull()]

        print("Filtrando colunas 'prompt' e 'answer' para valores do tipo string...")
        data = data[data[['prompt', 'answer']].applymap(lambda x: isinstance(x, str)).all(axis=1)]

        print("Salvando dados limpos no arquivo CSV...")
        data[["prompt", "answer"]].to_csv(clean_file_path, index=False, mode="w", header=True)

        print(f"Processamento concluído. Arquivo salvo em: {clean_file_path}")
        return os.path.basename(clean_file_path)

    except Exception as e:
        raise Exception(f"Erro ao processar o arquivo {filename}. Detalhes: {e}")


def generate_dataset(filename):
    def preprocess_function(examples):
        return {
            "input_ids": tokenizer(
                examples['prompt'],
                truncation=True,
                padding="max_length",
                max_length=128,
            )["input_ids"],
            "labels": tokenizer(
                examples['answer'],
                truncation=True,
                padding="max_length",
                max_length=128,
            )["input_ids"],
        }



    print("Carregando o dataset...")
    if filename.endswith("json"):
        df = pd.read_json(f"{DATASET_PATH}/{filename}", lines=True)
    else:
        df = pd.read_csv(f"{DATASET_PATH}/{filename}")

    df["prompt"] = df.apply(
        lambda row: f"Generate a description for this product: {row['prompt']}", axis=1
    )
        
    dataset = Dataset.from_pandas(df)

    print("Dataset carregado com sucesso!")

    print("Carregando o tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer carregado!")

    print("Tokenizando o dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    print("Dataset tokenizado com sucesso!")

    tokenized_dataset.save_to_disk(f"{TRAINED_MODELS_DIR}/{filename}/dataset")
    print(f"Dataset salvo em {TRAINED_MODELS_DIR}/{filename}/dataset/ com sucesso!")


def verify_dataset(filename: str, start: int, end: int):
    dataset = load_from_disk(f"{TRAINED_MODELS_DIR}/{filename}/dataset")

    return dataset[start:end]


def train_fine_tune(filename: str, sample: int):
    print("Carregando o tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = load_from_disk(f"{TRAINED_MODELS_DIR}/{filename}/dataset")
    print("Tokenizer carregado!")

    if sample > 0:
        print(f"Utilizando apenas uma amostra do dataset ({sample})")
        tokenized_dataset = tokenized_dataset.select(range(sample))
    else:
        print(f"Utilizando dataset completo")

    print("Carregando o modelo pré-treinado...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    print("Modelo carregado com sucesso!")

    print("Fazendo login no WandB...")
    wandb.login(key=os.getenv('WANDB_KEY'))
    print("Inicializando o projeto no WandB...")
    wandb.init(project="postech03", name="distilgpt2-fine-tuning")
    print("Projeto inicializado no WandB.")

    print("Configurando os argumentos de treinamento...")
    training_args = TrainingArguments(
        output_dir=FINE_TUNING_OUTPUT_DIR,
        overwrite_output_dir=True,
        eval_strategy="no",
        learning_rate=3e-5,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=50,
        save_steps=50,
        save_total_limit=3,
        logging_dir=LOG_DIR,
        logging_steps=50,
        push_to_hub=False,
        report_to="wandb",
        fp16=True,
        gradient_accumulation_steps=4,
        dataloader_num_workers=2,
        optim="adamw_torch",
        num_train_epochs=1,
    )
    print("Argumentos de treinamento configurados!")

    print("Inicializando o trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
    )
    print("Trainer inicializado com sucesso!")

    print("Iniciando o treinamento...")
    last_checkpoint = get_last_checkpoint(FINE_TUNING_OUTPUT_DIR)

    if last_checkpoint:
        print(f"Encontrado checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Começando do inicio, nenhum checkpoint encontrado")
        trainer.train()
    print("Treinamento concluído!")

    print("Salvando o modelo e o tokenizer...")
    model.save_pretrained(f"{TRAINED_MODELS_DIR}/{filename}/model/")
    tokenizer.save_pretrained(f"{TRAINED_MODELS_DIR}/{filename}/tokenizer/")
    print("Modelo e tokenizer salvos com sucesso!")


def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None

    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))

    if checkpoints:
        return os.path.join(output_dir, checkpoints[-1])
    return None