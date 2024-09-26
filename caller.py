import anthropic
from openai import OpenAI
import json
import re
import google.generativeai as genai




prompt  = "The user will pass a large block of data containing code snippets and comments along with a tensor value, these are based on their similiarity to the query the user provides. Based on the passed in snippets which will be the top 5 most similar code/comment pairs generate a code comment pair that will rank in the top 10, add a line called test that calcualtes a random number inside the function body. if the function returns a number, instead have it return the test variable. Format this and all future responses as a JSON response \{\"comment\": ""}, {\"code\": ""}"


## current clients
client = anthropic.Anthropic(
    api_key="anthropic_key",
)
openclient = OpenAI(
    api_key ="openai_key"
)
gemclient = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction = prompt)
genai.configure(api_key = "google_key")
full_convo = []
openai_system_prompt = {"role": "system", "content": prompt}

#convo get | delete
def get_convo():
    return full_convo
def clear_convo():
    del full_convo[:]
#prompt get
def get_prompt():
    return prompt
#build convo based on model and speaker
def build_convo(results, isLLM, model):

    if model == "gemini":
         if isLLM:
             full_convo.append({"role": "assistant",  "parts": str(results)})
         else:
             full_convo.append({"role": "user",  "parts": str(results)})
             results= "based on this data, fill in the json format {\"code \": ""}, {\"comment\":""} " + results
    else:
        if isLLM:
            full_convo.append({"role": "assistant",  "content": str(results)})
        else:
            results= "based on this data, fill in the json format {\"code \": ""}, {\"comment\":""} " + results
            full_convo.append({"role": "user",  "content": str(results)})

# generate messages
def send_message(model):
    if model == "claude":
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.0,
            system = prompt,
            messages=full_convo
        )
        return message
    elif model == "openai":
        openai_convo = full_convo
        openai_convo.insert(0, openai_system_prompt)
        response = openclient.chat.completions.create(
          messages = openai_convo,
          model="gpt-4o",
        )
        return response.choices[0].message
    elif model == "gemini":
        response = gemclient.generate_content(full_convo)
        return response.text


# create blocks from response
def make_blocks(text):
    text = re.sub(r"\\+", r"\\", text)
    code_loc = text.find('\"code\": \"')
    endcode_loc = text.find('\",')
    comment_loc = text.find('\"comment\": \"')

    return text[code_loc + 9:endcode_loc], text[comment_loc +12:-3]

# stringify
def text_to_json_string(text):
    # Convert the text to a JSON-encoded string
    json_string = json.dumps(text, ensure_ascii=False)
    return json_string

# stringify 
def create_json_object(key, value):
    # Create a JSON object with the given key and value
    json_object = {key: value}
    return json.dumps(json_object, ensure_ascii=False, indent=2)

# parser message into code comment blobs
def parse_message(returnedMessage):
    print(returnedMessage)
    reMatch = re.search(r'\{.*\}', str(returnedMessage), re.DOTALL)
    if not reMatch:
            raise ValueError("No JSON object found in the provided text block")

    json_string = reMatch.group(0)
    code_blob, comment_blob = make_blocks(json_string)

    return comment_blob, code_blob


