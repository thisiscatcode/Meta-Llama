from flask import Flask, request, jsonify
from llama_cpp import Llama
import mysql.connector
import time

app = Flask(__name__)

# Initialize Llama models
llm_jp = Llama(model_path="/llama.cpp/models/ELYZA-japanese-Llama-2-7b-instruct-q2_K.gguf",
               n_ctx=20480)
llm = Llama(model_path="/llama.cpp/models/llama-2-7b-chat.Q2_K.gguf",
            n_ctx=20480)

def get_llama_response_en(prompt):
    output = llm(
        prompt,
        max_tokens=-1,
        temperature=0.1,
        echo=True,
    )
    response = output["choices"][0]["text"]
    response_parts = response.split("### Response:")
    return response_parts[1].strip()

def get_llama_response_jp(prompt):
    output = llm_jp(
        prompt,
        max_tokens=-1,
        temperature=0.1,
        echo=True,
    )
    response = output["choices"][0]["text"]
    response_parts = response.split("### Response:")
    return response_parts[1].strip()

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_question = data.get('question')
    from_code = data.get('from_code')
    
    # Prepare the prompt
    prompt = f"###Instruction:{user_question}\n### Response:"
    
    start_time = time.time()
    
    if from_code == 'en' or from_code == 'Elyza':
        response = get_llama_response_jp(prompt) if from_code == 'Elyza' else get_llama_response_en(prompt)
        translated_response = response
    else:
        response = get_llama_response_en(prompt)
        translated_response = response  # Note: Translation should be handled by translation service
        
    total_execution_time = time.time() - start_time
    save_to_mysql(f"({from_code}2){user_question}", user_question, response, 
                 translated_response, total_execution_time)
    
    return jsonify({"response": translated_response})

@app.route('/get_response_batch', methods=['POST'])
def get_response_batch():
    data = request.json
    user_question = data.get('question')
    question_id = data.get('question_id')
    from_code = data.get('from_code')
    
    prompt = f"###Instruction:{user_question}\n### Response:"
    
    start_time = time.time()
    
    if from_code == 'en' or from_code == 'Elyza':
        response = get_llama_response_jp(prompt) if from_code == 'Elyza' else get_llama_response_en(prompt)
        translated_response = response
    else:
        response = get_llama_response_en(prompt)
        translated_response = response
        
    total_execution_time = time.time() - start_time
    save_to_mysql_batch(question_id, user_question, response, translated_response, 
                       total_execution_time)
    
    return jsonify({"response": translated_response})

def save_to_mysql(user_question, translated_english, llama_response, translated_japanese, 
                 total_execution_time):
    try:
        insert_query = """
            INSERT INTO bot_llama 
            (user_question, translated_english, llama_response, translated_japanese, 
             exe_time, creator, created_at) 
            VALUES (%s, %s, %s, %s, %s, %s, now())
        """
        
        data_to_insert = (user_question, translated_english, llama_response, 
                         translated_japanese, total_execution_time, "test")

        mysql_conn = mysql.connector.connect(
            host='xxx',
            user='xxx',
            password='xxx',
            database='xxx',
            charset='utf8mb4'
        )
        
        mysql_cursor = mysql_conn.cursor()
        mysql_cursor.execute(insert_query, data_to_insert)
        mysql_conn.commit()

    finally:
        if 'mysql_cursor' in locals():
            mysql_cursor.close()
        if 'mysql_conn' in locals():
            mysql_conn.close()

def save_to_mysql_batch(question_id, translated_english, llama_response, 
                       translated_japanese, total_execution_time):
    try:
        update_query = """
            UPDATE bot_llama 
            SET translated_english = %s, llama_response = %s, 
                translated_japanese = %s, exe_time = %s 
            WHERE id = %s
        """
        
        data_to_update = (translated_english, llama_response, translated_japanese, 
                         total_execution_time, question_id)

        mysql_conn = mysql.connector.connect(
            host='xxx',
            user='xxx',
            password='xxx',
            database='xxx'
        )
        
        mysql_cursor = mysql_conn.cursor()
        mysql_cursor.execute(update_query, data_to_update)
        mysql_conn.commit()

    finally:
        if 'mysql_cursor' in locals():
            mysql_cursor.close()
        if 'mysql_conn' in locals():
            mysql_conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
