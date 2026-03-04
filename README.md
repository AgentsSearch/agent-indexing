### build db and index
Just run the main.py file to execute the entire pipeline. It will initialize the database, build the index from the provided JSON file, and then run a test search query. You can modify the test search query in main.py to try out different searches!


### run api 
python -m uvicorn api:app --reload

### test response to use
curl -X POST http://127.0.0.1:8000/search \
-H "Content-Type: application/json" \
-d '{"query": "I need a tool to track expenses"}'



## For Probing
- pip install requirements.txt
- run main.py (modify  location of mcp agent files in main.py if needed)
- run api:" python -m uvicorn api:app --reload"
- see branch with changes to probing pipelines run_pipeline.py

