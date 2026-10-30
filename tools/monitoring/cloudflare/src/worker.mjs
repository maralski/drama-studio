// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0

const METRICS = new Set([
  "app.launch", "generation.succeeded", "generation.failed",
  "generation.cancelled", "setup.failed",
]);
const ENDPOINTS = {
  US: "https://metric-api.newrelic.com/metric/v1",
  EU: "https://metric-api.eu.newrelic.com/metric/v1",
  JP: "https://metric-api.jp.nr-data.net/metric/v1",
};
const MAX_BODY = 512;
const INTERVAL = 60_000;
const MAX_REQUESTS_PER_BATCH = 500;

function reply(status) {
  return new Response(null, { status, headers: { "Cache-Control": "no-store" } });
}

async function ingestKey(env) {
  if (!Object.hasOwn(ENDPOINTS, env.NEW_RELIC_REGION)) return null;
  const key = await env.NEW_RELIC_INGEST_KEY?.get();
  return typeof key === "string" && key.trim().length > 0 ? key : null;
}

export async function readCounters(request) {
  if (request.headers.get("Content-Type") !== "application/json"
      || request.headers.has("Content-Encoding") || !request.body) {
    throw new Error("Invalid payload");
  }
  const reader = request.body.getReader();
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY) throw new Error("Invalid payload");
      chunks.push(value);
    }
  } finally {
    await reader.cancel();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const input = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("Invalid payload");
  const entries = Object.entries(input);
  if (!entries.length || entries.some(([name, value]) =>
    !METRICS.has(name) || !Number.isInteger(value) || value < 1 || value > 10_000)) {
    throw new Error("Invalid payload");
  }
  return Object.fromEntries(entries);
}

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);
      if (url.search) return reply(400);
      if (url.pathname === "/health" && request.method === "GET") {
        if (!await ingestKey(env)) return reply(503);
        return await env.USAGE.get(env.USAGE.idFromName("global"))
          .fetch("https://usage.internal/health");
      }
      if (url.pathname !== "/v1/usage") return reply(404);
      if (request.method !== "POST") return reply(405);
      if (!await ingestKey(env)) return reply(503);
      let counters;
      try { counters = await readCounters(request); } catch { return reply(400); }
      const batch = env.USAGE.get(env.USAGE.idFromName("global"));
      // Construct a new request: no incoming headers, URLs or client metadata survive.
      return await batch.fetch("https://usage.internal/", {
        method: "POST", body: JSON.stringify(counters),
      });
    } catch {
      return reply(503);
    }
  },
};

export class UsageBatch {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
  }

  async fetch(request) {
    if (request.method === "GET" && new URL(request.url).pathname === "/health") {
      return Response.json({ delivery: (await this.ctx.storage.get("delivery")) ?? null },
        { headers: { "Cache-Control": "no-store" } });
    }
    const counters = await request.json();
    const accepted = await this.ctx.storage.transaction(async (storage) => {
      let batch = await storage.get("batch");
      if (!batch) {
        batch = { started: Date.now(), requests: 0, counts: {} };
        await storage.setAlarm(batch.started + INTERVAL);
      }
      if (batch.requests >= MAX_REQUESTS_PER_BATCH) return false;
      batch.requests += 1;
      for (const [name, count] of Object.entries(counters)) {
        batch.counts[name] = (batch.counts[name] ?? 0) + count;
      }
      await storage.put("batch", batch);
      return true;
    });
    return reply(accepted ? 204 : 429);
  }

  async alarm() {
    // Remove before sending: alarm retries must never replay a delivered batch.
    const batch = await this.ctx.storage.transaction(async (storage) => {
      const current = await storage.get("batch");
      if (!current) return null;
      // A repeated alarm must not flush a newly opened batch early.
      if (Date.now() < current.started + INTERVAL) {
        await storage.setAlarm(current.started + INTERVAL);
        return null;
      }
      await storage.delete("batch");
      return current;
    });
    if (!batch) return;
    const payload = [{ metrics: Object.entries(batch.counts).map(([name, value]) => ({
      name: `drama_studio.${name}`, type: "count", value,
      timestamp: Math.floor(batch.started / INTERVAL) * 60,
      "interval.ms": INTERVAL,
    })) }];
    let accepted = false;
    try {
      const key = await ingestKey(this.env);
      if (!key) throw new Error("Ingest unavailable");
      const response = await fetch(ENDPOINTS[this.env.NEW_RELIC_REGION], {
        method: "POST",
        headers: { "Content-Type": "application/json", "Api-Key": key },
        body: JSON.stringify(payload),
        redirect: "manual",
        signal: AbortSignal.timeout(10_000),
      });
      accepted = response.status === 202;
      await response.body?.cancel();
      // Delivery is best effort. Neither responses nor failures are logged or retried.
    } catch {}
    await this.ctx.storage.put("delivery", {
      accepted, minute: Math.floor(Date.now() / INTERVAL) * 60,
    });
  }
}
