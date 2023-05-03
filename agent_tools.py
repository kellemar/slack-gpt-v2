# Agent setup
from langchain.agents import tool
from langchain.utilities import GoogleSerperAPIWrapper

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
# Coingecko Linkup
from pycoingecko import CoinGeckoAPI
import time
import cache

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain

from langchain.chat_models import ChatOpenAI

import requests
import os
import re

from bs4 import BeautifulSoup


def remove_script_tags(text):
  pattern = r'<script[^>]*>.*?</script>'
  clean_text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
  return clean_text


# Agent Tools
@tool("code_snippets", return_direct=True)
def code_snippets(query: str) -> str:
  """Useful for generating code snippets. The input should be the query in natural language."""
  template = """Q:{question}"""
  prompt = PromptTemplate(template=template, input_variables=["question"])
  
  # Use GPT-4 since it's superior to GPT3.5 in terms of code completion.
  llm_chain = LLMChain(prompt=prompt,
                       llm=ChatOpenAI(model_name="gpt-4",
                                      temperature=0.7,
                                      max_tokens=8000),
                       verbose=True)

  response = llm_chain.predict(question=query)
  return response


@tool("company_knowledge", return_direct=False)
def company_knowledge(query: str) -> str:
  """Useful when questions are asked about Company Name. Use this more than Google Search. The input should be a question in natural language that this API can answer."""
  cached_key = query.lower().replace(" ", "_")
  cached = cache.get_cache(cached_key)
  if cached:
    #print(cached_video)
    return cached
  embeddings = OpenAIEmbeddings()
  persist_directory = 'db'
  vectorstore = Chroma(persist_directory=persist_directory,
                           embedding_function=embeddings,
                           collection_name="company_name")

  db_retrieval_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}))
  response = db_retrieval_chain(query)
  cache.set_cache(cached_key, response)
  print(response)
  return response

# TODO: Largely untested and not finished. Some issues with scraping certain sites.
@tool("scrape_website", return_direct=False)
def scrape_website(query: str) -> str:
  """Useful for accessing the contents of a website. The input should be the url of the website."""
  st = time.time()
  browserless_url = "https://chrome.browserless.io/content?token=" + os.environ.get(
    "BROWSERLESS_TOKEN")
  result = requests.post(url=browserless_url,
                         json={
                           "gotoOptions": {
                             "timeout": 10000,
                             "waitUntil": "networkidle4"
                           },
                           "url": query,
                           "elements": [{
                             "selector": "body"
                           }]
                         },
                         headers={"Content-Type": "application/json"})
  try:
    json_result = result
    cleaned_json = remove_script_tags(json_result)
    soup = BeautifulSoup(cleaned_json, 'html.parser')
    inner_text = soup.get_text()
    print("End of scrape_website: " + str(time.time() - st) + " seconds")
  except Exception as e:
    inner_text = "Unable to process website."
  return inner_text


@tool("web3_token_price", return_direct=False)
def web3_token_price(query: str) -> str:
  """Useful for when you want to ask questions about price, market cap and market volume of cryptocurrencies. The input should be the name of the coin or token."""
  st = time.time()
  token_price_header = "Game, Token, Token Price, Token Market Cap, Token Market Volume\n"
  game_list = query.split(",")
  print(game_list)
  token_stats = ""
  for g in game_list:
    token_stats += get_token_price(g)
  print(token_stats)
  print("End of Web3 Token Price Search: " + str(time.time() - st) +
        " seconds")
  return token_price_header + token_stats


def get_token_price(token_name: str) -> str:
  full_stat = ""
  try:
    cached = cache.get_cache(token_name)
    if cached:
      return cached
    token_symbol = token_name
    cg = CoinGeckoAPI()

    token_data = cg.search(token_symbol.lower())
    if token_data:
      # print("Token Data: " + str(token_data))
      if len(token_data['coins']) > 0:
        token_api_symbol = token_data['coins'][0]['api_symbol']
        token_stats = cg.get_price(ids=token_data['coins'][0]['api_symbol'],
                                   vs_currencies='usd',
                                   include_market_cap=True,
                                   include_24hr_vol=True)

        if token_stats:
          token_mcap = "{:,.0f}".format(
            token_stats[token_api_symbol]['usd_market_cap'])
          token_volume = "{:,.0f}".format(
            token_stats[token_api_symbol]['usd_24h_vol'])
          token_price = "{:,.2f}".format(token_stats[token_api_symbol]['usd'])
          full_stat = f"{token_symbol}, USD{token_price}, USD{token_mcap}, USD{token_volume}"
          cache.set_cache(token_name, full_stat)
          return f"{full_stat}\n"
  except Exception as e:
    print(f"web3_token_price:{e}")
    return f"No token prices available for {token_name}.\n"

  return f"No token prices available for {token_name}.\n"


@tool("google_search", return_direct=False)
def google_search(query: str) -> str:
  """Useful for when you need to answer questions about current events. The input is the search query."""
  search = GoogleSerperAPIWrapper()
  response = search.run(query)
  return response
