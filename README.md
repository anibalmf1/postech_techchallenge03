# Projeto FAST API - Treinamento de GPT2-Medium para Geração de Conteúdo

Este projeto é uma aplicação desenvolvida utilizando **FastAPI** que tem como objetivo treinar um modelo GPT2-Medium
para geração de textos baseados em títulos de produtos. Ele fornece uma interface de API para interagir com o modelo e
gerar conteúdos automaticamente.

## Estrutura do Projeto

O código principal do projeto está localizado no arquivo `main.py`, e a aplicação é configurada usando o framework *
*FastAPI**. Utilizamos o modelo `gpt2-medium` da biblioteca Hugging Face para realizar o treinamento e a geração de
textos.

## Pré-requisitos

Antes de executar este projeto, você precisará garantir que todos os requisitos de software estejam instalados no seu
ambiente:

- **Python 3.10 ou superior**
- **Pip** (gerenciador de pacotes do Python)

### Instalação das Dependências

Siga os passos abaixo para instalar as dependências do projeto.

1. Clone este repositório:

   ```bash
   git clone https://github.com/anibalmf1/postech_techchallenge03.git
   cd postech_techchallenge03
   ```

2. Instale as dependências listadas no arquivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Como Executar

Para rodar a aplicação, utilizamos o servidor de desenvolvimento `uvicorn`. Certifique-se de estar no diretório raiz do
projeto e faça o seguinte:

1. Inicie o servidor FastAPI com o comando:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

    - **`main`**: Refere-se ao nome do arquivo Python principal (`main.py`).
    - **`app`**: É a instância do FastAPI definida no arquivo `main.py`.

## Funcionamento do Projeto

- A API utiliza o framework **FastAPI** para expor rotas que permitem treinar o modelo GPT2-Medium e gerar textos.
- O modelo é carregado utilizando a biblioteca Hugging Face Transformers e pode ser ajustado com novos dados fornecidos
  pelo usuário.

### Exemplos de Uso

Aqui estão os exemplos de como utilizar os endpoints para realizar as principais operações da API:

#### Passo 1: Limpar Arquivo

Primeiro, você precisa limpar o arquivo original utilizando o endpoint `POST /train/clean` com o seguinte payload:

```json
{
  "filename": "trn.json"
}
```

Após a execução, um novo arquivo com os dados limpos será gerado e retornado como resultado.

#### Passo 2: Gerar o Dataset

Agora, gere o dataset a partir do arquivo limpo utilizando o endpoint `POST /train/dataset`:

```json
{
  "filename": "trn.cleaned.csv"
}
```

#### Passo 3: Validar o Dataset

Para garantir que o dataset foi gerado corretamente, valide-o com o endpoint
`POST /train/dataset/verify?start=0&end=10`:

```json
{
  "filename": "trn.cleaned.csv"
}
```

Esse processo verifica uma amostra do dataset com base nos índices de início (`start`) e fim (`end`) fornecidos.

#### Passo 4: Treinar o Modelo

Depois de validar o dataset, treine o modelo usando o endpoint `POST /train/model`. Este endpoint utiliza o arquivo
gerado como base para o treinamento do modelo GPT2-Medium:

```json
{
  "filename": "trn.cleaned.csv",
  "sample": 3000
}
```

> **Nota:** O campo `sample` é opcional. Ele pode ser usado para validar o modelo durante o treinamento com uma parte do
dataset. Recomenda-se treinar o modelo com o conjunto de dados completo para obter melhores resultados.

#### Passo 5: Gerar Conteúdo

Por fim, está na hora de gerar o conteúdo utilizando o endpoint `POST /generate`:

```json
{
  "prompt": "All about africa, 1st edition"
}
```

> **Nota:** O campo `filename` também é opcional neste endpoint. Se fornecido, ele carregará o modelo ajustado com o
dataset fornecido. Caso contrário, será usado o modelo pré-treinado original.

### Desenvolvimento e Testes

Durante o desenvolvimento, o servidor pode ser executado com a flag `--reload` (como mostrado acima), o que permite
recarregar automaticamente o código sempre que ele for alterado. Para testes, recomenda-se o uso de ferramentas como *
*Postman** ou **cURL**.
