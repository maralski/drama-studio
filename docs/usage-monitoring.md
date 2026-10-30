# Optional usage monitoring

The onboarding asks users to share anonymous usage information to help improve
Drama Studio. Neither button is preselected; Return chooses Don't Share. The
choice is separate from the model license. Existing installs are asked once too.
Settings > General > Privacy lets users change their choice. No collection or
request occurs before opt-in. Unconfigured builds never send requests.

## Data contract

The app accepts only five counters, with no additional attributes:

| Counter | Meaning |
| --- | --- |
| `app.launch` | App starts with sharing already enabled |
| `generation.succeeded` | A generation completes, including auditions and individual narrated sections |
| `generation.failed` | A generation fails, excluding cancellation |
| `generation.cancelled` | A running generation is stopped |
| `setup.failed` | A download, codec installation or tuning operation throws an error |

This covers the desktop app, not standalone CLI or HTTP API generation. It does
not install an auto-instrumentation, crash-reporting or session-replay agent.
Scripts, generated or recorded audio, voice names, file paths, error messages,
stack traces, hostnames, hardware details, and user/session/device identifiers
are excluded by the schema. There are no client timestamps or version dimensions.

Counts accumulate in memory and are sent at most once a minute. Each counter is
capped at 10,000 per batch. Requests use an ephemeral session without cookies,
credentials or caching. Redirects are refused. Failed batches are dropped without
retries; quitting can lose pending counts. Disabling sharing clears pending
counts and cancels an active request. Cancellation cannot retract a request that
has already reached the service. Re-enabling never replays prior activity.

## Cloudflare Workers relay

The deployable relay is in `tools/monitoring/cloudflare`. It accepts only the
counter schema at `POST /v1/usage`. It rejects unknown fields, query strings,
invalid values and bodies larger than 512 bytes, including streamed bodies.

One SQLite-backed Durable Object combines counters across all requests into
one-minute batches. It stores the combined counters, a server
start time, the batch request count, and whether the last delivery was accepted
with its server time rounded to a minute. The Worker sends a new Metric API request
with a rounded server timestamp and no attributes. It never forwards incoming
headers, IP addresses, URLs or other request metadata. New Relic sees a Cloudflare
connection, not the app's connection.

The batch is deleted before delivery. Failed delivery is discarded; retries of
an alarm cannot replay a sent batch. This can lose metrics during an outage or a
crash, by design. Stored aggregate counts survive Worker restarts until their
alarm runs; Cloudflare's storage recovery mechanisms may retain older aggregate
states. No individual request records or identifiers are stored.

The Worker accepts at most 500 requests per batch, then returns 429. This bounds
accepted counts and outgoing requests but is not DDoS protection: rejected traffic
still consumes Cloudflare requests. Anyone can submit plausible counts to this
public endpoint. Do not use the metrics for billing or security controls.

`wrangler.jsonc` disables Workers Logs, invocation logs, traces, Logpush and preview
URLs. There are no console logs, tail consumers or analytics integrations. Cloudflare
still handles client IPs at its network edge and may process operational/security
metadata under its own policies. These settings do not disable Cloudflare's own
platform processing. Avoid adding request logging or analytics in the dashboard.
This is not a guarantee against traffic analysis or correlations at low usage.

The configuration uses SQLite-backed Durable Objects, which are available on
[Cloudflare's free plan](https://developers.cloudflare.com/durable-objects/platform/pricing/).
Worker requests, Durable Object requests, execution time and storage operations
have separate quotas. Stay on Workers Free; when a quota is exhausted, collection
can fail until it resets. No paid services or custom domain are required.

### Deploy

Use Node.js 22 or later. Run these commands from the repository root:

```sh
cd tools/monitoring/cloudflare
npm ci
npm test
npm run check
npm run login
```

The login opens Cloudflare's browser authorization flow. Select the intended
account; if more than one is available, set `CLOUDFLARE_ACCOUNT_ID` for deployment.
If Cloudflare asks you to choose a `workers.dev` subdomain, choose one in the
Workers dashboard. The Worker name defaults to `drama-studio-usage`.

This deployment targets Cloudflare account `dbe4b37d905c7d88049e15d666df4ed9`
and the US New Relic region. `NEW_RELIC_INGEST_KEY` binds the existing account secret
`drama-studio-newrelic-ingest-key` in store `8ee11a16d1504af58c9d46d87a823e30`.
Only these resource identifiers are in source control; the secret value is not.
The secret must have the Workers permission scope. Then:

```sh
npm run deploy
```

Deployment creates the Worker and its Durable Object binding/migration and binds
the existing Secrets Store entry. Do not copy the key to the app or local files.
Rotate its value in Cloudflare Secrets Store without rebuilding the app. The
Worker reads it through the binding's asynchronous `get()` method.

`npm run check` packages locally without deploying. `.dev.vars*`, `.env*`, build
output and dependencies are ignored by Git. An inaccessible/empty account secret or invalid region
makes both ingestion and `GET /health` return 503. A 200 response proves the secret can be read. Its `delivery` field is initially
null; after a batch it reports only `accepted` and a rounded server `minute`.
`accepted: true` means the last batch received HTTP 202 from New Relic, not that
it passed subsequent asynchronous validation. No payload or response body is
exposed in health checks.

### Verify and configure the app

The deployed HTTPS address is:

```sh
curl --fail -i https://drama-studio-usage.drama-studio-usage-relay.workers.dev/health
curl --fail -i https://drama-studio-usage.drama-studio-usage-relay.workers.dev/v1/usage \
  -H 'Content-Type: application/json' --data '{"app.launch":1}'
```

Expect 200 from health and 204 from ingestion. Wait at least one minute and check
the `drama_studio.app.launch` metric in New Relic. Confirm that no client metadata
appears. A 204 acknowledges local acceptance into a batch, not New Relic delivery.
Check `/health` after the minute to confirm `delivery.accepted` is true, then
check the metric in New Relic. Check `NrIntegrationError` if it fails New Relic's
asynchronous validation.

Packaged builds use this deployed URL by default. From the repository root:

```sh
tools/make_app.sh
```

This sets `DSUsageEndpoint` in the app's Info.plist. Override `DS_USAGE_ENDPOINT`
to use another relay, or set it to an empty string to package without monitoring.
Never embed a New Relic key in the app. Unpackaged SwiftPM builds and packages
without a URL do not upload even if the user opts in.
Onboarding and Settings describe the purpose, optional choice and data categories
without naming the collection method or destination.

New Relic's [Metric API contract](https://docs.newrelic.com/docs/data-apis/ingest-apis/metric-api/report-metrics-metric-api/)
defines the ingest payload and regional endpoints. The relay sends `count`
metrics with a 60-second interval and no attributes. Example query:

```sql
FROM Metric SELECT sum(drama_studio.generation.succeeded),
  sum(drama_studio.generation.failed) SINCE 1 day ago TIMESERIES
```

Validate a deployment with synthetic counters first. Inspect the metric attributes
in New Relic before configuring a release.
No live New Relic delivery is exercised by the local tests.

## Checks

```sh
swift test --filter 'AppSetupTests|AppUsageMonitoringTests|AppSessionTests'
cd tools/monitoring/cloudflare
npm ci
npm test
npm run check
```

The tests cover onboarding choices, existing installs, default-off collection,
persistence, revocation, cancellation, discarded pre-consent data, the outgoing
app payload, and Worker validation, concurrent aggregation, missing configuration, batch limits
and delivery failures in the local Workers runtime. Outbound requests are intercepted
by the test harness, so tests cannot send metrics to New Relic.
