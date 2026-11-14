# Speech over the HTTP API

The app serves an HTTP API on 127.0.0.1 (port 8780 by default, 8791 in a verify home). `drama serve` serves the same API without the app. Clients send text and get audio back, as a file or a stream.

## Sub-features

- `GET /v1/health` (no token), `GET /v1/models`, `GET /v1/voices`, `POST /v1/voices`.
- `POST /v1/audio/speech`: a WAV file, or with `"stream": true` chunked PCM.
- `GET /v1/audio/speech/stream`: server-sent events, ending in a `done` or `error` event.
- The live route, for text sent sentence by sentence.
- `GET /v1/takes/{id}`, and retakes by take id.
- Request fields: `input`, `voice`, `model` (engine), `language`, `seed`, `expressiveness`, `note`.
- Errors: `401` without the token, `400` for a bad voice, language or engine, `429` when a generation is running.

## How to get to it (user POV)

The API indicator at the right of the app's status line shows "API server listening on http://127.0.0.1:PORT". The token is in the home folder as `api-token`. From a terminal a client calls it with curl; README.md has the full reference under the HTTP API heading.

## Driving it with verify.sh

```sh
ds_home ming && ds_app            # or ds_serve, for the server without the app
ds_api /v1/models
ds_api /v1/audio/speech -D $DS_EVIDENCE/speech.headers -o $DS_EVIDENCE/speech.wav \
  -H 'Content-Type: application/json' -d '{"input":"Tickets go on sale at nine.","seed":1}'
grep -i "^HTTP\|^x-drama" $DS_EVIDENCE/speech.headers; afinfo $DS_EVIDENCE/speech.wav | grep duration
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:$DS_PORT/v1/voices     # 401 without the token
```

Proof: `HTTP/1.1 200 OK`, `X-Drama-Engine`, `X-Drama-Take-Id`, `X-Drama-Word-Coverage: recognized`, and `afinfo` reports the duration in `X-Drama-Seconds`. For an error case, save the status and the JSON body.

## Gotchas

- Test both servers when you change `Sources/DramaServer`: the app wraps it in `APIServerController`, and `drama serve` does not.
- One generation runs at a time. A request sent while the app window is generating is refused; check `busy` in `/v1/health` first.
- A streamed response has already sent `200` when a late failure happens, so read the `error` event or the truncated body, not the status.
- `X-Drama-Frames` and `X-Drama-Generated-Frames` exist only on Drama 3.
- `tools/api_smoke.sh` reads the token from the real library and expects Drama 3 in `/v1/models`. It does not fit a Ming verify home.
