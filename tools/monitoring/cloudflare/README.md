# Drama Studio usage relay

Cloudflare Worker for opt-in, anonymous usage counts. The New Relic ingest key
lives in Cloudflare Secrets Store, never in the app. Worker logs and traces are disabled.

See [deployment instructions and data contract](../../../docs/usage-monitoring.md#cloudflare-workers-relay).

```sh
npm ci
npm test
npm run check
```

After Cloudflare login and selecting the New Relic region in `wrangler.jsonc`:

```sh
npm run deploy
```

The Worker uses the existing `drama-studio-newrelic-ingest-key` account secret.
It rejects ingestion when the secret cannot be read. No real key
or Cloudflare account is needed for the local tests or deployment dry run.
