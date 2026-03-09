Docker Compose mounts these files into the `dev` container at `/run/secrets/<NAME>`.

Fill in the files you actually need for your current provider/runtime setup.
Blank files are ignored by the runtime secret loader.

Expected file names:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`
- `TELEGRAM_TOKEN`
- `BRAVE_SEARCH_API_KEY`
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_AI_WORKERS_REST_API_KEY`
