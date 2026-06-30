from pathlib import Path

arquivos = [
    "modelos/floresta_isolacao.joblib",
    "modelos/svm_uma_classe.joblib"
]

for arquivo in arquivos:
    tamanho = Path(arquivo).stat().st_size
    print(f"{arquivo}:")
    print(f"  {tamanho} bytes")
    print(f"  {tamanho/1024:.2f} KB")
    print(f"  {tamanho/(1024**2):.2f} MB\n")