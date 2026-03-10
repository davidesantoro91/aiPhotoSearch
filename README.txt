comando per creare l'environment ->python -m venv .venv
comando per attivare l'environment -> source .venv/bin/activate   # su Windows: .venv\Scripts\activate
----------------------------
se nel comando precedente otteniamo un errore , possiamo agire in :
MODO 1) Sblocco temporaneo (solo per questa sessione PowerShell) -> Set-ExecutionPolicy Bypass -Scope Process -Force .\.venv\Scripts\Activate.ps1
MODO 2) Sblocco per l’utente corrente (persistent -> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser .\.venv\Scripts\Activate.ps1
		Chiudi/riapri PowerShell dopo il comando.
MODO 3) Non attivo la venv e usiamo il Python della venv direttamente 
	python -m venv .venv
	.\.venv\Scripts\python -m pip install -r requirements.txt
	.\.venv\Scripts\python -m streamlit run app.py
----------------------------

comando per installare i vari pacchetti contenuti dentro il requirements -> pip install -r requirements.txt
comando per startare l'app -> streamlit run app.py
comando per avviare l'app con il suo ambiente -> .\.venv\Scripts\python -m streamlit run appV1.py


