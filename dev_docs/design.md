# Jarvis Agent Design

## Components

The agent should have below components, some of which can be straightforward, some however I haven't fully decided what they'll be like exactly. As said, this is inspired by OpenClaw but not a exactly Python copy of it. The installable Python package now lives under `src/jarvis/`, and these components each live as subpackages or resource directories inside that package to keep the source modular and maintainable:

- `src/jarvis/llm/` . for now just a simple llm_client abstraction layer that handles request out and response in from AI providers (with token streaming), also with embedding models (later might use for vector searches). right now just add openai as a provider. NOTE: this layer should provide a normalized and standardized model output to the core agent loop, stripping out all privide-specific quirks. the agent core loop should expect a fixed standard model output from this AI provider service layer including model's response text, any parsed out and schema-validated tool calls

- `src/jarvis/core/` . this is the agent loop. we're not using any agent framework here, we're constructing our own agent loop, which will use the LLM service layer from `src/jarvis/llm/`

- `src/jarvis/identities/` . this mainly contains a series of "starter context content" md files, possibly using a similar design as openclaw, which means things like REACTOR.md (persona, boundaries, tone, like SOUL.md for openclaw), USER.md (user preferences + address/how to speak), PROGRAM.md (operating rules like “how I run tasks, when I write memory, what I optimize for”, like AGENTS.md for openclaw), and ARMOR.md (security-practice overlay).

- `src/jarvis/gateway/` . this is the control surface of the agent system's connection to outside. use Starlette with a minimal design (websocket), also handles session routing

- `src/jarvis/memory/` . this is the agent's long term memory system. Here I will like use a more complex system than openclaw's MEMORY.md. This will likely include memory specific tools that agent can use to read/write/edit memory, separate from general tools in `src/jarvis/tools/`, but also could include non-agent-controlled memory infra like regular memory consolidation, reorganization, update, decay, etc. I might build this following the file memory and graph memory approach like in another project of mine: `~/Documents/Projects/Python/simple_agent_memory`, tho I haven't decided whether i should use sql wrapper or just use plain md files

- `src/jarvis/ui/` . this is where the connection to messaging apps are. for now only implement connection with telegram: user token verification, send to (streaming) user on telegram, receive message sent from user on telegram, etc.

- `src/jarvis/tools/` . this is a heavy department. this includes the default tools (in cli) for the agent to read, write, edit, copy, move, remove, etc. file in its dedicated workspace (here is the agent_world/ dir running in docker container), also tools like web search, web fetch, anything basic that the agent could need. There will be a **tool registry** (a documentation of the tools and also exposed to the agent so agent will know what tools are available and how to use them), **tool runtime** (if needed, many filesystem related tools come with linux bash inside the container, some other tools will need runtime to be used by the agent), and a **policy interface** (restricting certain tool uses, or confirmation from user needed. This is not the policy itself but the policy interface which user can use to decide what tools are restricted)

- `src/jarvis/storage/` . This is a utility service that stores all past conversation transcripts, metadata like prompt hash, tool calls, durations, token usage, etc. stored in JSONL.

**NOTE:** I need to clarify this otherwise it could be potentially confusing. EVERYTHING I mentioned here is part of the repo, they're code (mostly, unless they're prompts). for example, `src/jarvis/storage/` is the service that grabs the conversation and stores it, while the actual runtime transcripts live under `archive/transcripts/` in the agent workspace in the docker container. same goes for e.g. `src/jarvis/memory/`. An **exception** is `src/jarvis/identities/`. this contains the core prompts that the agent will need. So they are a part of the code repo where they are created, but the same files will also be copied to the agent workspace as `identities/` too, which are the actual files that agent will use during runtime

## Scattered Design Choices

Here I'll list (will be updated by me as we go) some design choices that will be built into the project. these are in no particular order or scope, I'm just writing down what comes to mind.

- Using telegram as the messaging app for the agent UI here, DM only, as in I as user chat with the agent via a single DM thread (no group chats or channels or whatever, just one DM like I'm talking to one single friend)
- Use openclaw style thread handling: one single chat thread, auto compact when near context limit, or user explicit `/new` command sent from telegram, which starts a new LLM context
- whenever a new session starts (including compacted sessions), always include the starter context content
