"""
app_svm_f.py

CAPTCHA comportamental em tempo real com o SVM de Uma Classe treinado
(trabalho_f/modelos/svm_f.joblib). Interface e avaliação em app_captcha_f.py.

Uso:
    python trabalho_f/scripts/app_svm_f.py
"""

from app_captcha_f import executar

if __name__ == "__main__":
    executar("svm")
