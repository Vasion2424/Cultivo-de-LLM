import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datos"
SEED = 42
random.seed(SEED)


def save_text(name, examples):
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)

    text = "\n\n".join(
        item.strip() for item in examples if item.strip()
    )

    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] {path}: {len(text.encode('utf-8')):,} bytes")


def prepare_oasst2():
    print("[DESCARGA] OpenAssistant OASST2 en español")

    ds = load_dataset(
        "bertin-project/oasst2_es_instruct_hf",
        split="train"
    )

    groups = defaultdict(list)

    for row in ds:
        messages = row.get("instruction", [])
        lines = []
        first_user = ""

        for message in messages:
            role = message.get("role", "")
            content = str(message.get("content", "")).strip()

            if not content:
                continue

            if role == "system":
                continue

            if role == "user":
                if not first_user:
                    first_user = content
                lines.append("Usuario: " + content)

            elif role == "assistant":
                lines.append("Asistente: " + content)

        if (
            first_user
            and len(lines) >= 2
            and lines[0].startswith("Usuario:")
            and lines[-1].startswith("Asistente:")
        ):
            key = " ".join(first_user.lower().split())

            # Todas las respuestas alternativas para una misma pregunta
            # permanecen juntas, evitando separar esa pregunta entre etapas.
            conversation = "\n".join(lines)
            if conversation not in groups[key]:
                groups[key].append(conversation)

    keys = list(groups)
    random.shuffle(keys)

    if len(keys) < 20:
        raise RuntimeError("Se encontraron muy pocas conversaciones.")

    # Reservar preguntas distintas para evaluación.
    n_val = max(2, int(len(keys) * 0.10))
    val_keys = keys[:n_val]
    training_keys = keys[n_val:]

    # Se limita el tamaño inicial del experimento.
    training_keys = training_keys[:4000]

    # Data1 y data2 tienen preguntas distintas: data2 será la nueva etapa.
    data1_keys = training_keys[::2]
    data2_keys = training_keys[1::2]

    val1_keys = val_keys[::2]
    val2_keys = val_keys[1::2]

    def flatten(selected_keys):
        result = []
        for key in selected_keys:
            result.extend(groups[key])
        random.shuffle(result)
        return result

    save_text("data1.txt", flatten(data1_keys))
    save_text("data2.txt", flatten(data2_keys))
    save_text("data1_val.txt", flatten(val1_keys))
    save_text("data2_val.txt", flatten(val2_keys))

    print(f"[INFO] Preguntas distintas de entrenamiento: {len(training_keys)}")
    print(f"[INFO] Preguntas reservadas para validación: {len(val_keys)}")


def download_json(filename):
    url = (
        "https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/"
        f"resolve/main/{filename}"
    )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SovereignMiniGPT/1.0"}
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_sqac(document):
    result = []

    for article in document.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")

            for qa in paragraph.get("qas", []):
                question = str(qa.get("question", "")).strip()
                answers = qa.get("answers", [])

                if not question or not answers:
                    continue

                answer_obj = answers[0]
                answer = str(answer_obj.get("text", "")).strip()

                if not answer:
                    continue

                start = answer_obj.get("answer_start", 0)

                try:
                    start = int(start)
                except (TypeError, ValueError):
                    start = 0

                start = max(0, min(start, len(context)))
                left = max(0, start - 35)
                right = min(len(context), start + len(answer) + 35)

                snippet = context[left:right].strip()

                example = (
                    "Usuario: Responde utilizando el contexto.\n"
                    f"Contexto: {snippet[:110]}\n"
                    f"Pregunta: {question[:100]}\n"
                    f"Asistente: {answer[:100]}"
                )

                result.append(example)

    return result


def prepare_sqac():
    print("[DESCARGA] SQAC: preguntas y respuestas en español")

    train = extract_sqac(download_json("train.json"))
    validation = extract_sqac(download_json("dev.json"))
    test = extract_sqac(download_json("test.json"))

    random.shuffle(train)

    save_text("data3.txt", train[:2500])
    save_text("data3_val.txt", validation[:400])
    save_text("data3_test.txt", test[:400])

    print(f"[INFO] SQAC entrenamiento disponible: {len(train)} ejemplos")
    print(f"[INFO] Validación disponible: {len(validation)} ejemplos")
    print(f"[INFO] Prueba final disponible: {len(test)} ejemplos")


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    prepare_oasst2()
    prepare_sqac()

    print("\n[TERMINADO] Los TXT reales están en la carpeta datos.")
    print("Consulta las licencias antes de redistribuir los corpus.")


if __name__ == "__main__":
    main()