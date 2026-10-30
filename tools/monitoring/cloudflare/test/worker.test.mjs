// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { Miniflare, convertV4MiniflareOptions } from "miniflare";

const config = JSON.parse(await readFile(new URL("../wrangler.jsonc", import.meta.url)));
const requests = [];
let upstreamStatus = 202;

function runtime(bindings = {}) {
  return new Miniflare(convertV4MiniflareOptions({
    modules: [
      { type: "ESModule", path: new URL("harness.mjs", import.meta.url).pathname },
      { type: "ESModule", path: new URL("../src/worker.mjs", import.meta.url).pathname },
    ],
    modulesRoot: new URL("..", import.meta.url).pathname,
    compatibilityDate: config.compatibility_date,
    durableObjects: { USAGE: { className: "TestUsageBatch", useSQLite: true } },
    bindings: { NEW_RELIC_REGION: "EU", TEST_KEY: "test-only-ingest-key", ...bindings },
    outboundService: async (request) => {
      requests.push({ url: request.url, headers: Object.fromEntries(request.headers), body: await request.json() });
      return new Response(null, { status: upstreamStatus,
        headers: upstreamStatus === 307 ? { Location: "https://untrusted.example.test/" } : {},
      });
    },
  }));
}

function post(mf, body, headers = {}) {
  return mf.dispatchFetch("https://relay.test/v1/usage", {
    method: "POST", headers: { "Content-Type": "application/json", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

async function batchStub(mf) {
  const namespace = await mf.getDurableObjectNamespace("USAGE");
  return namespace.get(namespace.idFromName("global"));
}

test("runtime validates requests, batches concurrent clients and forwards no identifiers", async () => {
  const mf = runtime();
  requests.length = 0;
  try {
    for (const body of [null, [], {}, { email: "private@example.test" },
      { "app.launch": 1, script: "private script" }, { "app.launch": true },
      { "app.launch": 1.5 }, { "app.launch": 0 }, { "app.launch": 10001 },
      "{".repeat(513), '{"__proto__":{"secret":"private"}}']) {
      assert.equal((await post(mf, body)).status, 400);
    }
    assert.equal((await post(mf, { "app.launch": 1 }, { "Content-Encoding": "gzip" })).status, 400);
    assert.equal((await mf.dispatchFetch("https://relay.test/v1/usage?email=private")).status, 400);
    assert.equal((await mf.dispatchFetch("https://relay.test/unknown")).status, 404);
    assert.equal((await mf.dispatchFetch("https://relay.test/v1/usage")).status, 405);
    assert.equal((await mf.dispatchFetch("https://relay.test/health")).status, 200);
    const responses = await Promise.all(Array.from({ length: 10 }, () => post(mf, {
      "app.launch": 1, "generation.succeeded": 2,
    }, { Cookie: "identity=private", "User-Agent": "private-host", "X-Forwarded-For": "192.0.2.1" })));
    assert.ok(responses.every(response => response.status === 204));
    const stub = await batchStub(mf);
    const state = await (await stub.fetch("https://internal/test/state")).json();
    assert.equal(state.requests, 10);
    assert.deepEqual(state.counts, { "app.launch": 10, "generation.succeeded": 20 });
    assert.deepEqual(Object.keys(state).sort(), ["counts", "requests", "started"]);
    assert.equal(requests.length, 0, "no individual upload before the batch closes");
    await stub.fetch("https://internal/test/flush");
    assert.equal(requests.length, 1);
    const health = await (await mf.dispatchFetch("https://relay.test/health")).json();
    assert.equal(health.delivery.accepted, true);
    assert.deepEqual(Object.keys(health.delivery).sort(), ["accepted", "minute"]);
    const sent = requests[0];
    assert.equal(sent.url, "https://metric-api.eu.newrelic.com/metric/v1");
    assert.equal(sent.headers["api-key"], "test-only-ingest-key");
    for (const header of ["cookie", "x-forwarded-for", "cf-connecting-ip", "authorization"]) {
      assert.equal(sent.headers[header], undefined);
    }
    assert.ok(!JSON.stringify(sent).includes("private"));
    assert.deepEqual(sent.body, [{ metrics: [
      { name: "drama_studio.app.launch", type: "count", value: 10, timestamp: 1790000040, "interval.ms": 60000 },
      { name: "drama_studio.generation.succeeded", type: "count", value: 20, timestamp: 1790000040, "interval.ms": 60000 },
    ] }]);
    assert.equal(await (await stub.fetch("https://internal/test/state")).json(), null);
    await stub.fetch("https://internal/test/alarm");
    assert.equal(requests.length, 1, "repeated alarms do not duplicate delivery");
    await post(mf, { "app.launch": 1 });
    await stub.fetch("https://internal/test/alarm");
    assert.equal(requests.length, 1, "late alarm does not flush a new batch early");
  } finally { await mf.dispose(); }
});

test("missing key or invalid region disables ingestion", async () => {
  for (const bindings of [{ TEST_KEY: "" }, { NEW_RELIC_REGION: "invalid" }, { TEST_SECRET_FAILURE: true }]) {
    const mf = runtime(bindings);
    try {
      assert.equal((await post(mf, { "app.launch": 1 })).status, 503);
      assert.equal((await mf.dispatchFetch("https://relay.test/health")).status, 503);
    } finally { await mf.dispose(); }
  }
});

test("batch rate limit bounds ingestion and failed delivery is discarded", async () => {
  const mf = runtime();
  requests.length = 0;
  upstreamStatus = 503;
  try {
    for (let i = 0; i < 500; i++) assert.equal((await post(mf, { "setup.failed": 1 })).status, 204);
    assert.equal((await post(mf, { "setup.failed": 1 })).status, 429);
    const stub = await batchStub(mf);
    await stub.fetch("https://internal/test/flush");
    assert.equal(requests.length, 1);
    assert.equal(requests[0].body[0].metrics[0].value, 500);
    assert.equal((await (await mf.dispatchFetch("https://relay.test/health")).json()).delivery.accepted, false);
    await stub.fetch("https://internal/test/alarm");
    assert.equal(requests.length, 1);
    assert.equal(await (await stub.fetch("https://internal/test/state")).json(), null);
    assert.equal((await post(mf, { "setup.failed": 1 })).status, 204);
    upstreamStatus = 307;
    await stub.fetch("https://internal/test/flush");
    assert.equal(requests.length, 2, "redirects never forward the key to another host");
    assert.equal(requests[1].url, "https://metric-api.eu.newrelic.com/metric/v1");
  } finally { upstreamStatus = 202; await mf.dispose(); }
});

test("deployment disables logs and traces and contains no test handlers or credentials", async () => {
  assert.equal(config.main, "src/worker.mjs");
  assert.equal(config.observability.enabled, false);
  assert.equal(config.observability.logs.enabled, false);
  assert.equal(config.observability.logs.invocation_logs, false);
  assert.equal(config.observability.traces.enabled, false);
  assert.equal(config.logpush, false);
  assert.equal(config.preview_urls, false);
  assert.equal(config.vars.NEW_RELIC_LICENSE_KEY, undefined);
  assert.equal(config.vars.NEW_RELIC_INGEST_KEY, undefined);
  assert.equal(config.secrets_store_secrets[0].secret_name, "drama-studio-newrelic-ingest-key");
  assert.deepEqual(config.migrations[0].new_sqlite_classes, ["UsageBatch"]);
});
