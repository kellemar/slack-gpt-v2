import openai
import os
import time
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

import traceback

# Langchain setup
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Agents
from langchain import LLMChain, PromptTemplate
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import agent_tools

#JSON
import json

import image_generator

# Configuration
# These are used for labelling the queries with specific intents
INTENTS = """market cap", "volume", "token", "price", "availability", "i want to create image or photo", "i want to create code", "i want to create social media", "download"""

DEFAULT_PROMPT = (
  "You are a friendly assistant called 'ChatBot', for a company that can answer general questions. Your goal is to help the people in "
  "the company with any questions they might have.  If you aren't sure about something, you should say that you don't know."
)

ADDITIONS = """Only answer in the following JSON structure {"answer": "..","subject_matter": [".."],"intents": [".."]} For the "intents" field, choose from the following values that best describes the intents: """ + INTENTS + "\n\nContext:\n"

# The OpenAI model to use. Can be gpt-3.5-turbo or gpt-4.
MODEL = os.environ.get("MODEL", "gpt-3.5-turbo")
MODEL_4 = "gpt-4"
# The max length of a message to OpenAI.
MAX_TOKENS = 8000 if MODEL == "gpt-4" else 4096
# The max length of a response from OpenAI.
MAX_RESPONSE_TOKENS = 1000
# Starts with "sk-", used for connecting to OpenAI.
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# Starts with "xapp-", used for connecting to Slack.
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
# Starts with "xoxb-", used for connecting to Slack.
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
# Tokens are ~4 characters but this script doens't account for that yet.
TOKEN_MULTIPLIER = 4

# Initialize the Slack Bolt App and Slack Web Client
app = App()
slack_client = WebClient(token=SLACK_BOT_TOKEN)

# Set up the default prompt and OpenAI API
prompt = os.environ.get("PROMPT", DEFAULT_PROMPT) + ADDITIONS
openai.api_key = OPENAI_API_KEY
llm_chat = ChatOpenAI(model_name=MODEL,
                      temperature=0.5,
                      max_tokens=MAX_RESPONSE_TOKENS)
llm_chat_4 = ChatOpenAI(model_name=MODEL_4, temperature=0.5)

# Tools for a General Agent
tools = load_tools(["wikipedia"], llm=llm_chat)
tools.pop(0)
tools.append(agent_tools.google_search)
tools.append(agent_tools.web3_token_price)
tools.append(agent_tools.code_snippets)
#tools.append(agent_tools.scrape_website)


# Tools for an Agent specifically for searching through a document store
company_tools = load_tools(["wikipedia"], llm=llm_chat)
company_tools.pop(0)
company_tools.append(agent_tools.avg_knowledge)
company_tools.append(agent_tools.google_search)


def generate_completion_langchain(prompt, messages, query):
  print("Query:" + query)
  query_context = get_context(query)
  if 'intents' in query_context:
    if 'i want to create image or photo' in query_context['intents']:
      #To handle any wrong queries being sent to image generation
      try:
        return image_generator.generate_image(query)
      except:
        return generate_completion_google(prompt, messages, query)
    if any(s in query.lower()
           for s in ("company_name","company_name2")):
      return generate_completion_company(prompt, messages, query)

  return generate_completion_google(prompt, messages, query)


# To generate a completion from the Google Agent
def generate_completion_google(prompt, messages, query):
  google_agent = initialize_agent(
    tools,
    llm_chat_4,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=messages)
  completion = google_agent.run(input=DEFAULT_PROMPT + query)
  return completion

# To generate a completion from the Document Store Agent
def generate_completion_company(prompt, messages, query):
  print("Query:" + query)
  company_agent = initialize_agent(
    company_tools,
    llm_chat,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=messages)
  completion = company_agent.run(input=query)
  return completion

# Get context of query, and label with intents
def get_context(query: str) -> str:
  output = {}
  template = """Answer only in JSON format: \
  {{"answer": "..", intents":[up to 3 intents that closely matches the query]}} \
  """ + INTENTS + \
    "\nQuestion: {question}"
  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt,
                       llm=ChatOpenAI(model_name="gpt-3.5-turbo",
                                      temperature=0),
                       verbose=True)

  response = llm_chain.predict(question=query)
  print(response)
  try:
    output = json.loads(response)
  except Exception as e:
    print(e)
    output['intents'] = []
    output['subject_matter'] = {}
  return output


def create_help_response():
  json_help = [{
    "type": "section",
    "text": {
      "type":
      "mrkdwn",
      "text":
      "Hello! To use me in the way possible, be as detailed as you can when you want to ask questions."
    }
  }, {
    "type":
    "actions",
    "block_id":
    "actionblock789",
    "elements": [
      {
        "type": "button",
        "text": {
          "type": "plain_text",
          "text": "Examples of Questions"
        },
        "action_id": "button_examples",
        "style": "primary",
        "value": "click_me_456"
      },
      {
        "type": "button",
        "text": {
          "type": "plain_text",
          "text": "Asking about the Company"
        },
        "action_id": "button_company_example",
        "value": "click_me_456"
      },
      {
        "type": "button",
        "text": {
          "type": "plain_text",
          "text": "How to use the image generator"
        },
        "action_id": "button_create_image",
        "value": "click_me_456"
      },
    ]
  }]
  return json_help


def get_message_history(channel_id, user_id, event_ts, limit, thread=False):
  """Fetch conversation or thread history and build a list of messages."""
  history = []
  memory = ConversationBufferMemory(memory_key="chat_history",
                                    return_messages=True)

  # Fetch the message history
  if thread:
    result = slack_client.conversations_replies(channel=channel_id,
                                                ts=event_ts,
                                                limit=limit,
                                                latest=int(time.time()))
  else:
    result = slack_client.conversations_history(channel=channel_id,
                                                limit=limit)

  token_count = 0

  for message in result["messages"]:
    #print("\nOriginal Message: " + str(message))
    if message.get("user") == user_id:
      role = "user"

    # Check for image generation from the bot
    elif ("files" in message and message["files"][0]["title"]
          ) == "AI Image Generator" or message.get(
            "subtype") == "bot_message" or message.get("bot_id"):
      role = "assistant"
    else:
      continue

    # Ignore typing responses from the bot
    if message["text"] == "Typing a response...":
      continue
    token_count += len(message["text"])
    if token_count > (MAX_TOKENS - MAX_RESPONSE_TOKENS):
      break
    else:
      try:

        if role == "user":
          memory.chat_memory.add_user_message(message["text"])
        else:
          memory.chat_memory.add_ai_message(message["text"])
        history.append({"role": role, "content": message["text"]})
      except Exception as e:
        traceback.print_exc()
        print(message["text"])

  # DMs are in reverse order while threads are not.
  if not thread:
    history.reverse()
  return memory, history


def handle_message(event, thread=False):
  """Handle a direct message or mention."""
  channel_id = event["channel"]
  user_id = event["user"]
  event_ts = event["ts"]
  completion_message = " "
  # Set up the payload for the "Typing a response..." message
  payload = {"channel": channel_id, "text": "Typing a response..."}

  if thread:
    # Use the thread_ts as the event_ts when in a thread
    event_ts = event.get("thread_ts", event["ts"])
    payload["thread_ts"] = event_ts

  # Get message history
  chat_history, history = get_message_history(channel_id,
                                              user_id,
                                              event_ts,
                                              limit=4,
                                              thread=thread)
  actual_query = history[-1]['content']
  # Send "Typing a response..." message
  typing_message = slack_client.chat_postMessage(**payload)

  # Generate the completion
  try:
    # This is a help command, send instructions back
    if actual_query.lower() == "help":
      completion_message = create_help_response()
      slack_client.chat_update(channel=channel_id,
                               text=" ",
                               ts=typing_message["ts"],
                               blocks=completion_message)
      return

    else:
      completion_message = generate_completion_langchain(
        prompt, chat_history, actual_query)
  except Exception as e:
    traceback.print_exc()
    completion_message = (
      "Something happened when trying to answer your query. Please try again.")

  # Show Image Generation response with image file
  if 'File:' in completion_message:
    slack_client.chat_delete(channel=channel_id, ts=typing_message["ts"])
    split_message = completion_message.split("File:")
    slack_client.files_upload_v2(channel=channel_id,
                                 title="AI Image Generator",
                                 file=split_message[1],
                                 initial_comment=split_message[0])

  else:
    slack_client.chat_update(channel=channel_id,
                             ts=typing_message["ts"],
                             text=completion_message)


def send_slack_message(response, channel_id):
  slack_client.chat_postMessage(channel=channel_id, text=response)


def send_block_message(response, channel_id):
  slack_client.chat_postMessage(channel=channel_id, text=" ", blocks=response)


@app.event("app_mention")
def mention_handler(body, say):
  """Handle app mention events."""
  event = body["event"]
  handle_message(event, thread=True)


@app.event("message")
def direct_message_handler(body, say):
  """Handle direct message events."""
  event = body["event"]
  print(body)
  if event.get("subtype") == "bot_message" or event.get("bot_id"):
    return
  handle_message(event)


@app.action("button_examples")
def handle_examples(ack, body, logger):
  ack()

  # TODO: More interactive examples
  '''
  json_response = [{
    "type": "section",
    "text": {
      "type": "mrkdwn",
      "text": "*Examples of questions:*"
    }
  }, {
    "type": "section",
    "text": {
      "type": "mrkdwn",
      "text": "*What's the price of Bitcoin?*"
    },
    "accessory": {
      "type": "button",
      "text": {
        "type": "plain_text",
        "text": "Try this"
      },
      "value": "example_query_1",
      "action_id": "buttons_examples_questions"
    }
  }, {
    "type": "section",
    "text": {
      "type": "mrkdwn",
      "text": "*What is Quake3 about?*"
    },
    "accessory": {
      "type": "button",
      "text": {
        "type": "plain_text",
        "text": "Try this"
      },
      "value": "example_query_2",
      "action_id": "buttons_examples_questions"
    }
  }, {
    "type": "section",
    "text": {
      "type": "mrkdwn",
      "text": "*eth marketcap?*"
    },
    "accessory": {
      "type": "button",
      "text": {
        "type": "plain_text",
        "text": "Try this"
      },
      "value": "example_query_3",
      "action_id": "buttons_examples_questions"
    }
  }]
  send_block_message(json_response, body['channel']['id'])
'''
  send_slack_message(
      "*Some Example Questions:*\n\
What is Quake 3?\n\
BTC Price?\n\
Tell me about how blockchain transactions work", body['channel']['id'])

    

@app.action("button_create_image")
def handle_examples(ack, body, logger):
  ack()
  print(body)
  send_slack_message(
    "\n*How to use the image generator:*\nStart your question similar to the phrase\n _create an image_ \n*followed your request in the form of these examples. The more descriptive, the better your images will turn out:*\n\
\n\n*Something simple:*\ncan you create an image of a dog playing with a ball in the forest\
\n\n*More complex prompts:*\ngenerate a photo of pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, art by winona nelson \
\n\nI want to create an image of ghost inside a hunted room, art by lois van baarle and loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image \
\n\nGenerate an image of red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation, huge scene, grass, art greg rutkowski \
\n\nI want to create an ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski",
    body['channel']['id'])


@app.action("button_company_example")
def handle_company_queries(ack, body, logger):
  ack()
  print(body)
  send_slack_message(
    "*Company example Questions include context such as Company name:*\n\
Who is Mr Boss?\n\
What is our roadmap for Company Name?\n\
What is Company Name?", body['channel']['id'])


if __name__ == "__main__":
  handler = SocketModeHandler(app, SLACK_APP_TOKEN)
  handler.start()
