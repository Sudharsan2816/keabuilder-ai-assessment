# KeaBuilder AI Engineer — System Design Answers

---

## Q1: Lead Classification System Design

### Classification Logic

The system uses Claude `claude-sonnet-4-20250514` with a structured prompt that enforces three tiers:

| Tier | Signals | Response Tone |
|------|---------|---------------|
| **HOT** | Pricing asked, deadline mentioned, specific use case, budget stated, "ready to start" | Urgent, direct, push for next step |
| **WARM** | Exploring options, general questions, "looking for tools", no urgency | Nurturing, educational, trust-building |
| **COLD** | One-word messages, no business context, "just curious", missing contact + no message | Light, curious, non-pushy |

### Classification Prompt (exact)

```
You are a lead qualification AI for KeaBuilder, a funnel and lead capture SaaS platform.

Given a lead form submission, you must:
1. Classify the lead as HOT, WARM, or COLD
2. Generate a personalized, human-sounding response
3. Handle incomplete inputs with exactly 1 clarifying question

CLASSIFICATION RULES:
HOT  → Clear intent + urgency + specific need
WARM → Genuine interest but vague timeline or unclear need
COLD → No clear need, just browsing, very generic, incomplete input

OUTPUT FORMAT:
Return ONLY valid JSON:
{
  "classification": "HOT",
  "confidence": 0.92,
  "reasoning": "one sentence explaining why",
  "signals": ["signal1", "signal2", "signal3"],
  "response": "personalized reply to the lead",
  "follow_up_question": "question if input unclear, else null"
}
```

### Human-feeling responses

- Name is injected if provided: `"Hey Ravi!"` vs generic opener
- Pain point is reflected back: `"A product launch funnel with a one-month runway..."`
- Each tier has a distinct emotional register — urgency for HOT, curiosity for COLD
- No template filler phrases ("I hope this email finds you well")

### Handling incomplete inputs

If key fields are missing (name, email, business_type), the model detects them from the message context. If the message itself is vague (e.g., "hi"), the model:
- Classifies as COLD
- Sets `follow_up_question` to a single targeted question
- Does NOT assume anything about their business

### Sample Input → Output

**Input:**
```json
{
  "name": "Ravi Kumar",
  "email": "ravi@startup.com",
  "business_type": "SaaS",
  "message": "Need funnels for product launch next month. Pricing?",
  "source": "landing_page"
}
```

**Output:**
```json
{
  "classification": "HOT",
  "confidence": 0.94,
  "reasoning": "Lead has a concrete deadline (next month), specific use case (product launch), and explicitly asked for pricing.",
  "signals": ["deadline mentioned: next month", "pricing explicitly requested", "specific use case: product launch"],
  "response": "Hey Ravi! A product launch funnel with a one-month runway is exactly where KeaBuilder shines. I'd love to walk you through our pricing and show you a template that fits SaaS launches — we can typically have you live in 48 hours. Are you free for a 15-minute call this week?",
  "follow_up_question": null
}
```

---

## Q2: Multi-Provider Content Routing System

### Architecture Diagram

```
Builder UI (React)
       │
       │ POST /generate { type: "image"|"video"|"voice", prompt, userId }
       ▼
┌──────────────────────┐
│   API Gateway        │  ← Rate limiter (100 req/min per user)
│   + Auth Middleware  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Content Router      │  ← Strategy Pattern dispatch
│  (Node.js service)   │
└──────┬───────┬───────┘
       │       │       │
    image    video   voice
       │       │       │
  StabilityAI  RunwayML  ElevenLabs
  / DALL-E 3  / Pika Labs / Play.ht
       │       │       │
       └───────┴───────┘
                │
                ▼
         S3 Output Storage
         /users/{userId}/{type}/{jobId}.ext
                │
                ▼
         CDN (CloudFront)
                │
                ▼
      WebSocket → notify UI
```

### Routing Logic (pseudocode)

```javascript
const PROVIDERS = {
  image: new StabilityAIProvider(),   // fallback: DallE3Provider
  video: new RunwayMLProvider(),      // fallback: PikaLabsProvider
  voice: new ElevenLabsProvider()     // fallback: PlayHtProvider
}

async function generateContent(type, payload) {
  const provider = PROVIDERS[type]
  if (!provider) throw new Error('Unknown content type: ' + type)

  const jobId = uuid()
  await db.jobs.create({ jobId, userId: payload.userId, status: 'processing' })

  try {
    const result = await generateWithFallback(provider, payload)
    const s3Path = `users/${payload.userId}/${type}/${jobId}.${result.ext}`
    await s3.upload(s3Path, result.buffer)
    await db.jobs.update(jobId, { status: 'complete', url: cdnUrl(s3Path) })
    await ws.notify(payload.userId, { jobId, status: 'complete', url: cdnUrl(s3Path) })
  } catch (err) {
    await db.jobs.update(jobId, { status: 'failed', error: err.message })
    await ws.notify(payload.userId, { jobId, status: 'failed' })
  }

  return { jobId }  // immediate return — UI polls or waits for WS
}
```

### Frontend Builder UI Interaction

```
1. User clicks "Generate Image" in Builder
2. UI sends: POST /generate { type: "image", prompt: "...", userId }
3. API returns immediately: { jobId: "abc-123" }
4. UI enters Optimistic State (spinner + "Generating your image...")
5. WebSocket receives: { jobId: "abc-123", status: "complete", url: "https://cdn.../image.png" }
6. UI renders final output and replaces spinner
```

No polling needed if WebSocket is active. Fallback: `GET /jobs/{jobId}` every 3 seconds.

### Storage Path Structure

```
S3 Bucket: keabuilder-assets/
  └── users/
      └── {userId}/
          ├── image/
          │   └── {jobId}.png
          ├── video/
          │   └── {jobId}.mp4
          └── voice/
              └── {jobId}.mp3
```

---

## Q3: LoRA Integration for Personalised AI Images

### Full Pipeline

```
Step 1: User uploads 5–15 reference photos (face, brand logo, product shots)
         → Stored in S3: /lora/{userId}/training-images/

Step 2: Training job triggered (async):
         - Base model: stabilityai/sdxl-base-1.0
         - Method: DreamBooth-LoRA
         - Steps: 1000–1500
         - Learning rate: 1e-4
         - LoRA rank: 4–8 (default 4)
         - Trigger word assigned: "sks"

Step 3: Trained weights saved:
         → S3: /lora/{userId}/adapter.safetensors

Step 4: At inference, user's LoRA loaded on top of base SDXL:
         → Trigger word injected into every prompt

Step 5: Output image returned via CDN
```

### Inference Code (Python, diffusers)

```python
from diffusers import StableDiffusionXLPipeline
import torch

def generate_with_lora(user_id: str, user_prompt: str, s3_client) -> bytes:
    lora_path = f"/tmp/lora/{user_id}/adapter.safetensors"
    s3_client.download_file("keabuilder-assets", f"lora/{user_id}/adapter.safetensors", lora_path)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.load_lora_weights(lora_path)

    image = pipe(
        prompt=f"photo of sks person, {user_prompt}",
        negative_prompt="blurry, distorted, low quality",
        num_inference_steps=30,
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": 0.85}
    ).images[0]

    return image
```

### How User Uploads Trigger LoRA Loading

```
Upload Event → S3 trigger → Lambda:
  1. Check if /lora/{userId}/adapter.safetensors exists in S3
  2. If yes: load at inference time (no re-training needed)
  3. If new images uploaded: queue DreamBooth training job
  4. Notify user when training complete (WebSocket / email)
```

### Trade-offs

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| **Rank** | 4 = fast training, smaller file, slight quality loss | 16 = better quality, slower, larger weights |
| **Scale** | 0.5 = subtle influence | 1.0 = strong override (can cause artifacts) |
| **Steps** | 500 = fast, underfits | 2000 = overfit risk |
| **Sweet spot** | Rank 4–8, Scale 0.75–0.85, Steps 1000–1500 | |

---

## Q4: Face / Text Similarity Search System

### Storage Design

```
S3: Raw files (images, documents)
    keabuilder-assets/users/{userId}/uploads/{fileId}.{ext}

PostgreSQL + pgvector: Embeddings + metadata
    Table: assets          → file metadata
    Table: asset_embeddings → vector embeddings (768-dim)
```

### Full PostgreSQL Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE assets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL,
    file_type   VARCHAR(20) NOT NULL,  -- 'image', 'text', 'face'
    s3_path     TEXT NOT NULL,
    cdn_url     TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    metadata    JSONB DEFAULT '{}'
);

CREATE TABLE asset_embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id    UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    model       VARCHAR(100) NOT NULL,  -- 'all-MiniLM-L6-v2', 'insightface-r100'
    embedding   VECTOR(768),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON asset_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

### Retrieval SQL Query (pgvector)

```sql
SELECT
    a.id,
    a.s3_path,
    a.cdn_url,
    a.metadata,
    1 - (ae.embedding <=> $1::vector) AS similarity_score
FROM asset_embeddings ae
JOIN assets a ON ae.asset_id = a.id
WHERE a.user_id = $2
  AND ae.model = $3
ORDER BY ae.embedding <=> $1::vector
LIMIT $4;
```

Parameters: `$1` = query embedding, `$2` = userId, `$3` = model name, `$4` = top_k

### Matching Thresholds

| Score Range | Meaning | Action |
|-------------|---------|--------|
| > 0.85 | Strong match | Auto-suggest or auto-apply |
| 0.50–0.85 | Partial match | Show as suggestion with confidence % |
| 0.30–0.50 | Weak match | Show with disclaimer |
| < 0.30 | No match | Return empty or "no similar content found" |

### Text Similarity (Demo 2 reference)

Demo 2 (`demo2_similarity_search/`) implements TF-IDF cosine similarity for text.
Production upgrade replaces TF-IDF with `sentence-transformers/all-MiniLM-L6-v2` (semantic embeddings) stored in pgvector.
For face similarity: `InsightFace` or `DeepFace` generates 512-dim face embeddings, same pgvector query applies.

---

## Q5: Multi-AI Fallback Strategy

### 3-Layer Fallback Architecture

```
Request
   │
   ├─ Layer 1: Retry (same provider)
   │     ├─ Attempt 1 → fail → wait 1s
   │     ├─ Attempt 2 → fail → wait 2s
   │     └─ Attempt 3 → fail → escalate
   │
   ├─ Layer 2: Provider Fallback
   │     ├─ Primary provider (e.g., StabilityAI) failed all retries
   │     └─ Switch to backup provider (e.g., DALL-E 3)
   │           ├─ Attempt 1 → fail → wait 1s
   │           ├─ Attempt 2 → fail → wait 2s
   │           └─ Attempt 3 → fail → escalate
   │
   └─ Layer 3: Job Queue
         ├─ All providers failed
         ├─ Add to BullMQ / SQS with 60s delay
         ├─ Notify user: "Your content is queued, you'll be notified shortly"
         └─ Worker retries → delivers when successful
```

### Complete Implementation (JavaScript pseudocode)

```javascript
async function generateWithFallback(request) {
  const providers = [primaryProvider, backupProvider]

  for (const provider of providers) {
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const result = await withTimeout(provider.generate(request), 30_000)
        return { status: 'success', data: result }
      } catch (err) {
        console.warn(`Provider ${provider.name} attempt ${attempt} failed: ${err.message}`)
        if (attempt < 3) await sleep(attempt * 1000)  // 1s, 2s
      }
    }
    console.warn(`Provider ${provider.name} exhausted all retries`)
  }

  // All providers failed — queue it
  const jobId = await jobQueue.add('generate', request, {
    delay: 60_000,
    attempts: 5,
    backoff: { type: 'exponential', delay: 2000 }
  })

  await notifyUser(request.userId, {
    type: 'generation_queued',
    message: "We're working on it — you'll get a notification when your content is ready."
  })

  return { status: 'queued', jobId }
}

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), ms))
  ])
}
```

### UX Pattern — What the user sees

| Layer | What happens | User sees |
|-------|-------------|-----------|
| Layer 1 | Retry, same provider | Spinner continues — no interruption |
| Layer 2 | Switch provider | Spinner continues — "Using backup system..." toast |
| Layer 3 | All failed, queued | "We're processing your request — check back in a minute" banner |
| Success from queue | Job completes | Push notification / WebSocket: "Your content is ready!" |

Never expose raw error messages (API timeout, 503, etc.) to the end user.

---

## Q6: High-Volume AI Request Handling

### Full Architecture Diagram

```
Users (1000s concurrent)
       │
       ▼
┌─────────────────────────┐
│  CloudFront CDN         │  ← Static assets + cached GET responses
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  API Gateway            │  ← Rate limiter: 100 req/min per user
│  + WAF + Auth           │    Burst limit: 500 req/s global
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Redis Cache            │  ← Check: has this prompt been generated before?
│  (ElastiCache)          │    TTL: 1hr | Hit rate: ~35% at scale
└────────┬────────────────┘
         │ cache miss
         ▼
┌─────────────────────────┐
│  SQS / BullMQ Queue     │  ← Decouple request intake from processing
│  (request queue)        │    Visibility timeout: 60s
└────────────┬────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  Worker Pool (ECS / Lambda)          │
│  ┌────────┐ ┌────────┐ ┌────────┐   │  ← Auto-scale on queue depth > 50
│  │Worker 1│ │Worker 2│ │Worker N│   │    Min: 2 workers, Max: 50 workers
│  └────────┘ └────────┘ └────────┘   │
└──────────────┬───────────────────────┘
               │
               ▼
    AI Provider APIs (Anthropic, StabilityAI, etc.)
               │
               ▼
       S3 Output Storage → CloudFront CDN
               │
               ▼
    WebSocket Server → notify user's browser
```

### Queue + Worker Pool Pattern

```
Producer (API):  enqueue(jobId, payload)
Queue:           SQS FIFO or BullMQ (Redis-backed)
Consumer:        Workers pull jobs, process, ack/nack
Dead Letter:     Failed jobs → DLQ → alert + manual review
```

### Redis Caching Strategy

```python
cache_key = sha256(f"{type}:{prompt}:{model}:{resolution}")

cached = redis.get(cache_key)
if cached:
    return json.loads(cached)  # skip AI call entirely

result = await ai_provider.generate(request)
redis.setex(cache_key, 3600, json.dumps(result))  # 1hr TTL
return result
```

Cache hit for identical prompts (e.g., same landing page image regenerated) = zero AI cost.

### Auto-Scaling Trigger

```
CloudWatch Alarm:
  Metric: SQS ApproximateNumberOfMessagesVisible
  Threshold: > 50 messages
  Action: ECS scale-out → +10 workers (max 50 total)

Scale-in:
  Metric: < 5 messages for 5 minutes
  Action: scale-in → -5 workers (min 2 total)
```

### Cost Optimization Summary

| Strategy | Savings |
|----------|---------|
| Redis prompt cache | 30–40% API call reduction |
| CDN for outputs | Eliminates repeat S3 reads |
| Spot instances for workers | ~70% compute cost reduction |
| Batching (Anthropic Batch API) | 50% cost for non-realtime jobs |
| LoRA weights shared per user | Avoid re-downloading at each request |

---

## Q7: Tools and Frameworks Used

### This Assessment

| Component | Tool / Library | Reason |
|-----------|---------------|--------|
| API Framework | **FastAPI** (Python) | Async support, auto Swagger docs, Pydantic validation |
| AI Model | **Claude claude-sonnet-4-20250514** (Anthropic) | Best-in-class instruction following + JSON output |
| NLP / Similarity | **scikit-learn** TF-IDF + cosine similarity | Zero-dependency demo; production path documented |
| Data Validation | **Pydantic v2** | Strict type enforcement at API boundary |
| Environment Config | **python-dotenv** | 12-factor app config pattern |
| HTTP Runtime | **Uvicorn** (ASGI) | Production-grade async server for FastAPI |

### Production Additions (referenced in design)

| Component | Tool | Purpose |
|-----------|------|---------|
| Semantic embeddings | sentence-transformers | Replace TF-IDF for meaning-aware similarity |
| Vector DB | pgvector (PostgreSQL) | Store + query embeddings at scale |
| Image generation | diffusers (HuggingFace) | SDXL + LoRA inference |
| Job queue | BullMQ / SQS | Async AI request processing |
| Cache | Redis (ElastiCache) | Prompt deduplication |
| CDN | CloudFront | Output delivery |
| Face recognition | InsightFace / DeepFace | Face embedding generation |

### Background

Experience with Python data pipelines, ETL workflows, and API integration informs the architecture decisions in this assessment — particularly around queue-based processing, caching strategies, and graceful degradation patterns.

Both working demos (Demo 1: Lead Classifier, Demo 2: Similarity Search) serve as practical implementations of the concepts described in Q1 and Q4.

**Demo 1** proves: Claude API integration, structured output parsing, prompt engineering for tone variation.
**Demo 2** proves: vector similarity concepts, production upgrade path reasoning, knowledge base design.
