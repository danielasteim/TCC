
# PORTUGUÊS
# CAPTCHA Comportamental – Coleta de Dados

Este projeto tem como objetivo a **coleta de dados comportamentais de usuários** durante a interação com uma interface gráfica, visando a construção de um dataset para estudos de **detecção de humanos vs bots** utilizando técnicas de *One-Class Classification* (One-Class SVM e Isolation Forest) como parte do meu Trabalho de Conclusão de Curso.

Para me ajudar, basta possuir alguma versão de Python e alguma IDE der sua preferência, clonar o repositório, rodar e interagir com o sistema o máximo de vezes que puder, zipar a pasta 'captcha_data' com todos seus dados capturados e me enviar por e-mail/mensagem para contribuir com a construção orgânica do DATASET.

A aplicação registra métricas como movimentação do mouse, cliques e temporização, armazenando os dados em arquivos `.json` para posterior análise. Nenhum dado sensivél é capturado. Nenhum dado será divulgado publicamente ou usado para fins comerciais, será usado apenas para pesquisa.

### 📁 Estrutura do Projeto

```
.
├── data_collection.py
├── requirements.txt
├── captcha_data/
│   ├── session_<id>_<user>.json
│   ├── ...
└── README.md
```

A pasta `captcha_data/` é gerada automaticamente após a execução do programa.

### 🔧 Dependências

Será necessário ter instalado: 
* **Python 3.9 ou superior**
* **pip**
* **venv**
* **IDE de sua preferência**

As principais bibliotecas utilizadas incluem: `tkinter`, `json`, `time`, `uuid`, `os`

#### LINUX 

```bash
sudo apt update
sudo apt install python3-tk
git clone #link
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python data_collection.py
```

#### Windows (PowerShell)

```powershell
git clone #link
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
python data_collection.py
```

Ao executar, o sistema irá:

1. Solicitar a identificação do usuário (utilize seu primeiro nome e sobrenome, issp é apenas para contabilizar a quantia de humanos distintos). 
2. Exibir a interface gráfica do CAPTCHA comportamental.
3. Registrar todas as interações realizadas durante a sessão.
4. Salvar os dados automaticamente em um arquivo `.json` dentro da pasta `captcha_data/`.
5. Se possivél realize 100 sessões ou quantas puder. Cada sessão gera um arquivo `.json`.

### 📂 Envio dos Dados Coletados

Após finalizar a coleta, envie **a pasta completa `captcha_data/`**, contendo todos os arquivos `.json`.

### Opções de envio:

1. Compactar a pasta
2. Enviar via:

* Google Drive
* GitHub (repositório privado ou público)
* E-Mail: danieladesa01@gmail.com com o assunto "CAPTCHA".

Certifique-se de que **todos os arquivos `.json` estejam incluídos**.


### 📊 Observações Importantes

* Os dados coletados são utilizados **exclusivamente para fins acadêmicos**.
* Nenhuma informação sensível é armazenada.
* O identificador do usuário é usado apenas para diferenciação de sessões.


### 👩‍💻 Autora
Daniela de Sá Steim
Projeto desenvolvido como parte do Trabalho de Conclusão de Curso em Engenharia da Computação.
**Tema:** CAPTCHA Comportamental
**Técnicas:** One-Class SVM e Isolation Forest


