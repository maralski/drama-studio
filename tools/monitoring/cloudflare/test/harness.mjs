// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Test-only entry point. Wrangler deploys src/worker.mjs instead.
import worker from "../src/worker.mjs";

function testEnv(env) {
  return { ...env, NEW_RELIC_INGEST_KEY: {
    async get() {
      if (env.TEST_SECRET_FAILURE) throw new Error("synthetic secret access failure");
      return env.TEST_KEY;
    },
  } };
}

export default { fetch(request, env) { return worker.fetch(request, testEnv(env)); } };
import { UsageBatch } from "../src/worker.mjs";

export class TestUsageBatch extends UsageBatch {
  constructor(ctx, env) { super(ctx, testEnv(env)); }
  async fetch(request) {
    const path = new URL(request.url).pathname;
    if (path === "/test/state") {
      return Response.json((await this.ctx.storage.get("batch")) ?? null);
    }
    if (path === "/test/flush") {
      const batch = await this.ctx.storage.get("batch");
      if (batch) {
        batch.started = 1_790_000_040_000;
        await this.ctx.storage.put("batch", batch);
      }
      await this.ctx.storage.deleteAlarm();
      await this.alarm();
      return new Response(null, { status: 204 });
    }
    if (path === "/test/alarm") {
      await this.alarm();
      return new Response(null, { status: 204 });
    }
    return super.fetch(request);
  }
}
