"""
app_floresta_f.py

CAPTCHA comportamental em tempo real com a Floresta de Isolação treinada
(trabalho_f/modelos/floresta_f.joblib). Interface e avaliação em app_captcha_f.py.

Uso:
    python trabalho_f/scripts/app_floresta_f.py
"""

from app_captcha_f import executar

if __name__ == "__main__":
    executar("floresta")
