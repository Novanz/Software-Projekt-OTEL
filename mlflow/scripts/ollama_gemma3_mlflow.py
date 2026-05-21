import os
import mlflow
from mlflow.openai import autolog
from openai import OpenAI

# 1. Configure MLflow tracking server
mlflow.set_tracking_uri("http://192.168.56.103:5000")
mlflow.set_experiment("ollama-gemma3-tests")

# 2. Enable MLflow autologging for OpenAI-compatible clients
autolog()  # will capture prompts, responses, and metadata [web:172]

# 3. Configure OpenAI client to talk to Ollama
# Ollama uses an OpenAI-compatible API at /v1 with a dummy API key
os.environ["OPENAI_API_KEY"] = "dummy-key"
client = OpenAI(
    base_url="http://192.168.56.103:11434/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

model_name = "gemma3:4b"
prompt = (
    "In 4 concise bullet points, describe our current setup:\n"
    "- Fedora CoreOS VM with static host-only IP\n"
    "- Quadlet-managed Ollama running gemma3:4b\n"
    "- Quadlet-managed MLflow tracking server\n"
    "- Goal: local LLM experiments with proper logging"
)

with mlflow.start_run(run_name="gemma3-4b-openai-client-001"):
    # 4. Call Ollama via OpenAI chat completion API
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a concise technical explainer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )

    answer = response.choices[0].message.content

    # 5. Log a couple of extra things manually (optional)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("client", "openai-to-ollama")
    # mlflow.log_text(prompt, "prompt.txt")
    # mlflow.log_text(answer, "response.txt")

    print("Model answer:\n")
    print(answer)
