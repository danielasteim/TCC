# ENGLISH

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

# Behavioral CAPTCHA – Data Collection

This project aims to **collect behavioral user data** during interaction with a graphical interface, with the goal of building a dataset for **human vs. bot detection** using *One-Class Classification* techniques (One-Class SVM and Isolation Forest), as part of my **Undergraduate Thesis**.

To contribute, you only need **any Python version** and an **IDE of your choice**. Simply clone the repository, run the application, and interact with the system as many times as possible. After that, zip the `captcha_data` folder containing all captured data and send it to me via email or message to help with the organic construction of the dataset.

The application records metrics such as **mouse movement**, **click events**, and **timing information**, storing the data in `.json` files for later analysis. **No sensitive data is collected.** The data will **not be publicly disclosed nor used for commercial purposes**, being used exclusively for academic research.

## 📁 Project Structure

```
.
├── data_collection.py
├── requirements.txt
├── captcha_data/
│   ├── session_<id>_<user>.json
│   ├── ...
└── README.md
```

The `captcha_data/` folder is automatically generated after running the program

## 🔧 Dependencies

The following are required:

* **Python 3.9 or higher**
* **pip**
* **venv**
* **An IDE of your choice**

The main libraries used include:
`tkinter`, `json`, `time`, `uuid`, `os`

When running the application, the system will:

1. Ask for user identification (use your **first name and last name**; this is only used to count distinct human participants).
2. Display the behavioral CAPTCHA graphical interface.
3. Record all interactions performed during the session.
4. Automatically save the data into a `.json` file inside the `captcha_data/` folder.
5. If possible, complete **at least 100 sessions** or as many as you can. Each session generates a separate `.json` file.

## 📂 Sending the Collected Data

After finishing the data collection, send **the complete `captcha_data/` folder**, containing all `.json` files.

### Sending options:

1. Compress the folder
2. Send via:

   * Google Drive
   * GitHub (public or private repository)
   * Email: **[danieladesa01@gmail.com](mailto:danieladesa01@gmail.com)**
     **Subject:** `CAPTCHA`

Make sure that **all `.json` files are included**.

## 📊 Important Notes

* The collected data is used **exclusively for academic purposes**.
* No sensitive or personal information is stored.
* The user identifier is used only to distinguish between sessions.

## 👩‍💻 Author

**Daniela de Sá Steim**
Project developed as part of the Undergraduate Thesis in Computer Engineering.

**Topic:** Behavioral CAPTCHA
**Techniques:** One-Class SVM and Isolation Forest