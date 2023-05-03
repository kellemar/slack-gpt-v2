# Slack ChatGPT Bot with Google, Coingecko Price check, and Document QnA, etc.

This fork was inspired by Aaron Ng ([@localghost](https://twitter.com/localghost))._

Some additions have been made such as adding Google Search and CoinGecko support via Langchain Agents. Also added support for generating images via StabilityAI's Dreamstudio APIs. This also includes Document QnA via ChromaDB, or if you wish, Postgres PGVectors.

The new additions also include Redis Caching support. I've tried using the native LLM Cache, but it isn't clear on what the TTL was. So I decided to stick with Redis, and it gives me more customisation on the TTLs of each data type.

## Introduction

This script creates a Slack bot that uses ChatGPT to respond to direct messages and mentions in a Slack workspace. While you can add QnA, it currently includes connections
to Google, Coingecko and DreamStudio via Langchain Agents.

## Environment Variables

### Required:

1. `OPENAI_API_KEY`: Your OpenAI API key, which starts with "sk-".
2. `SLACK_APP_TOKEN`: Your Slack App Token, which starts with "xapp-".
3. `SLACK_BOT_TOKEN`: Your Slack Bot Token, which starts with "xoxb-".
4. `SERPER_API_KEY`: Your key to access google serper API on Langchain. Get it from serper.dev 
5. `REDIS_PASSWORD`: Redis cache password
6. `REDIS_URL`: URL to your Redis server
7. `STABILITY_KEY`: Stability AI key from Dreamstudio.ai
8. `STABILITY_HOST`: Host to use Stability AI's image generation servers
9. `PGVECTOR_CONNECTION_STRING`:"postgresql+psycopg2://username:password@db_url:5432/db_name"



### Optional:

1. `MODEL`: The OpenAI model to use. Can be "gpt-3.5-turbo" or "gpt-4". Default is "gpt-3.5-turbo".
2. `PROMPT`: A custom prompt for the bot. Default is a predefined prompt for a friendly company assistant.

## Setup

1. Go to [https://api.slack.com/apps?new_app=1](https://api.slack.com/apps?new_app=1).
2. Click "Create New App".
3. Click "Basic", then name your Slack bot and select a workspace.

### Configuration

1. In "Settings" → "Socket Mode", enable both Socket Mode and Event Subscriptions.
2. In "Settings" → "Basic Information", install your app to the workspace by following the instructions.
3. In "Settings" → "Basic Information", scroll to "App-Level Tokens" and create one with the permission `connections:write`. Set the resulting token that starts with `xapp-` as your `SLACK_APP_TOKEN`.
4. In "Features" → "OAuth and Permissions" → "Scopes", add the following permissions: `app_mentions:read`, `channels:history`, `channels:read`, `chat:write`, `chat:write.public`, `groups:history`, `groups:read`, `im:history`, `im:read`, `mpim:history`, `mpim:read`, `users:read`, `files.write`, `files.read`.
5. In "Features" → "Event Subscriptions" → "Subscribe to Bot Events", add the following bot user events: `app_mentions:read`, `message.im`.
6. In "Features" → "App Home", turn on the "Messages Tab" switch, and enable the `Allow users to send Slash commands and messages from the messages tab` feature.
7. In "Features" → "OAuth and Permissions", copy the "Bot User OAuth Token" and set it as the `SLACK_BOT_TOKEN` in your environment.


Now your Slack bot should be ready to use!

### Additional Configuration due to new additions:

1. Get the Serper keys from https://serper.dev. The key should replace values in SERPER_API_KEY in the .env file.
2. Get the Stability AI API Keys by signing up for https://beta.dreamstudio.ai/
3. Get the Redis credentials by creating your own instance, or create one from https://upstash.com


## Deployment

### Cloud Deployment:

1. If deploying to a cloud service, check out and reconfigure `setup.sh` and `start.sh`.

### Local Deployment:

1. If running locally, install dependencies with `poetry`.
2. Comment out these two lines in the script:

```
# from dotenv import load_dotenv
# load_dotenv()
```

Start the bot and enjoy using it in your Slack workspace.

### Ingesting Data:

There are 2 files that you can use to ingest the data needed.
ingest_chroma.py - Adds your data into a local ChromaDB and creates a /db/ folder which stores the indexes.

OR

ingest_pg.py - Adds your data into a Supabase Postgres PGVector-supported table. 

If you are using PGVectors, you will have update the vectorstore code in both main.py and agent_tools.py to support it.

### Caveats:
1. Agents are still somewhat unrealiable. You can probably make them better by better describing the prompts.
2. ValueErrors still happen sometimes for LLM Outputs. This can be resolved by querying the Bot again and catch via exception handling.