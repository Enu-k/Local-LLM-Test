
1-a.  Install Ollama (macOS example)
brew install ollama

1-b.  Start the Ollama daemon in another terminal
ollama serve &              # keeps running in background

1-c.  Pull a model (feel free to swap for any Ollama-3 compatible model)
ollama pull deepseek-r1:14b


Then open another terminal and run: 

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

TO execute: 
python3 --csv data/CSV_FILE_NAME.csv -- question "YOUR_QUESTION"


