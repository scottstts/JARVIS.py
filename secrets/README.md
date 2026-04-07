Docker Compose mounts these files into the `jarvis_runtime` container at `/run/secrets/<NAME>`.

Fill in the files you actually need for your current provider/runtime setup.
Blank files are ignored by the runtime secret loader.

Expected file names:

- `OPENAI_API_KEY`: https://platform.openai.com/api-keys
- `ANTHROPIC_API_KEY`: https://platform.claude.com/settings/keys
- `GOOGLE_API_KEY`: https://aistudio.google.com/api-keys
- `XAI_API_KEY`: https://console.x.ai/
- `OPENROUTER_API_KEY`: https://openrouter.ai/workspaces/default/keys
- `TELEGRAM_TOKEN`: https://core.telegram.org/bots/api
- `JARVIS_UI_TELEGRAM_ALLOWED_USER_ID`: https://core.telegram.org/bots/api
- `BRAVE_SEARCH_API_KEY`: https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
- `CLOUDFLARE_ACCOUNT_ID`: https://developers.cloudflare.com/workers-ai/
- `CLOUDFLARE_AI_WORKERS_REST_API_KEY`: https://developers.cloudflare.com/workers-ai/
- `SENDER_EMAIL_ADDRESS`: A gmail address for Jarvis to send emails with
- `SMTP_PASSWORD`: https://support.google.com/accounts/answer/185833?hl=en
