# Carlitos

Carlitos is Filippo's personal assistant inspired by Carlos Alcaraz (the tennis player) even if obviously Sinner is stronger.

## Running Carlitos...

### Create an mcp.json file in this format

```json
{
    "mcpServers": {
        "sequential-thinking": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-sequential-thinking"
            ],
            "description": "Handles multi-step reasoning tasks and complex problem solving requiring sequential thinking"
        },
        "Puppeteer": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-puppeteer"
            ],
            "description": "Provides web automation and browser interaction capabilities"
        },
        "gmail_composio": {
            "url": "",
            "description": "Manages Gmail operations including sending, reading, and organizing emails"
        },
        "calendar_composio": {
          "url" : "",
          "description": "Handles calendar operations including creating, updating, and scheduling events"
        },
        "notion_composio": {
            "url": "",
            "description": "Manages Notion workspace operations including pages, databases, and content"
        },
        "slack_composio": {
            "url": "",
            "description": "Facilitates Slack messaging and channel management operations"
        },
        "linear_composio": {
            "url": "",
            "description": "Manages Linear issue tracking and project management operations"
        }
    }
}
```

> Check in the FAQs how to generate the composio URLs for the integrations.

### Vamos Carlitos

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

> Remember to set the `OPENAI_API_KEY` environment variable in your .env file.

## FAQs

### What Carlitos can do?

- Handle the calendar
- Handle the emails

### What Carlitos will be able to do?

Imagination is the limit.

### How to create the correct MCP configuration file?

Better if you ask Filippo... or check out this video. In order to quickly integrate with all the providers, the easiest and quickest solution it's to use Composio. Composio exposes a MCP server for each of the providers.

### Which integrations does Carlitos support right now?

- Calendar
- Gmail
- Slack
- Notion
- Linear
- GitHub

### Why not the other integrations?

I don't care... Carlitos has been built with the only purpose of serving Filippo. If you want to add a new integration, you can do it by yourself.