import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "exploracao_capitais.ipynb"

DEFAULT_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}


def garantir_estrutura_minima(notebook):
    notebook.setdefault("cells", [])
    notebook.setdefault("metadata", {})
    notebook.setdefault("nbformat", 4)
    notebook.setdefault("nbformat_minor", 5)

    metadata = notebook["metadata"]
    for chave, valor in DEFAULT_METADATA.items():
        metadata.setdefault(chave, valor)

    for cell in notebook["cells"]:
        cell.setdefault("metadata", {})
        cell.setdefault("source", [])

        if cell.get("cell_type") == "code":
            cell.setdefault("execution_count", None)
            cell.setdefault("outputs", [])

    return notebook


def main():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Notebook nao encontrado em: {NOTEBOOK_PATH}\n"
            "Crie ou restaure o notebook antes de rodar este script."
        )

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook = garantir_estrutura_minima(notebook)

    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Notebook sincronizado a partir da versao atual: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
