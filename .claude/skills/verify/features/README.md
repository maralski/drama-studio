# Drama Studio feature map

One file per feature a user touches. Each says what the feature is, how a user reaches it, how to drive it with `verify.sh`, and what observable end state proves it. Read the file for the feature you changed, and drive every entry point it lists.

| Feature | File | Surfaces |
| --- | --- | --- |
| Generate a take from a script | [generate-take.md](generate-take.md) | app, Speech menu |
| Speech over the HTTP API | [api-speech.md](api-speech.md) | app's server, `drama serve` |
| Voices: built-in, library, designed | [voices.md](voices.md) | app Inspector, API, `drama-say` |
| Engine choice: Ming and Drama 3 | [engines.md](engines.md) | setup window, Inspector, Speech menu, API `model` |
| `drama-say` and the MCP tools | [command-line.md](command-line.md) | terminal |

Not mapped yet: document import and section narration, export of audio and captions, project save and reopen, recovery after a crash, pronunciation settings, the menu bar extra, app updates. Add a file when you first verify one.
