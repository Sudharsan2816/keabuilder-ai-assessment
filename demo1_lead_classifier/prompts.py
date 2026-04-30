CLASSIFY_PROMPT = """
You are a lead qualification AI for KeaBuilder, a funnel and lead capture SaaS platform.

Given a lead form submission, you must:
1. Classify the lead as HOT, WARM, or COLD
2. Generate a personalized, human-sounding response
3. Handle incomplete inputs with exactly 1 clarifying question

CLASSIFICATION RULES:
HOT  → Clear intent + urgency + specific need
       Signals: pricing asked, deadline mentioned, specific use case,
                budget mentioned, "ready to start", "need this now"

WARM → Genuine interest but vague timeline or unclear need
       Signals: exploring options, general questions,
                "looking for tools", no urgency

COLD → No clear need, just browsing, very generic, incomplete input
       Signals: one-word messages, no business context,
                "just curious", no contact info + no message

RESPONSE RULES:
- Use their name if provided, skip it if not
- Reference their specific business/pain point directly
- Sound like a real human, not a template
- HOT tone: urgent, direct, push for next step
- WARM tone: nurturing, educational, build trust
- COLD tone: light, curious, non-pushy, spark interest
- For incomplete inputs: ask exactly 1 question, do not assume

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown. No backticks. No explanation. Just JSON:
{
  "classification": "HOT",
  "confidence": 0.92,
  "reasoning": "one sentence explaining why this classification",
  "signals": ["signal1", "signal2", "signal3"],
  "response": "personalized reply to the lead",
  "follow_up_question": "question if input unclear, else null"
}
"""

SYSTEM_DESIGN = {
    "multi_provider_routing": {
        "description": "Strategy Pattern for routing content generation to different AI providers",
        "providers": {
            "image": "Stability AI / DALL-E 3",
            "video": "RunwayML / Pika Labs",
            "voice": "ElevenLabs / Play.ht"
        },
        "routing_logic": """
const PROVIDERS = {
  image: new StabilityAIProvider(),
  video: new RunwayMLProvider(),
  voice: new ElevenLabsProvider()
}

async function generateContent(type, payload) {
  const provider = PROVIDERS[type]
  if (!provider) throw new Error('Unknown content type: ' + type)
  return await provider.generate(payload)
}
        """,
        "storage": "S3 → /users/{userId}/{type}/{jobId}.{ext}",
        "frontend_pattern": "Optimistic UI → POST returns jobId → Poll GET /jobs/{jobId} → Render on complete"
    },
    "lora_integration": {
        "description": "DreamBooth-LoRA on SDXL for consistent face/branding",
        "steps": [
            "User uploads 5-15 reference images",
            "LoRA training triggered (1000-1500 steps, lr=1e-4, rank=4-8)",
            "Weights saved to S3: /lora/{userId}/adapter.safetensors",
            "At inference: load base SDXL + user LoRA weights",
            "Use trigger word in prompt: 'photo of sks person, {prompt}'"
        ],
        "inference_code": """
pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/sdxl-base-1.0')
pipe.load_lora_weights(s3_lora_path)
image = pipe(
    prompt=f'photo of sks person, {user_prompt}',
    cross_attention_kwargs={'scale': 0.85}
).images[0]
        """,
        "tradeoffs": "Rank 4=fast+small, Rank 16=quality+slow. Scale 0.7-0.85 sweet spot."
    },
    "fallback_strategy": {
        "description": "3-layer fallback for multi-AI service reliability",
        "layers": {
            "layer_1": "Retry with exponential backoff (1s, 2s, 4s delays)",
            "layer_2": "Provider fallback (primary fails → secondary)",
            "layer_3": "Job queue (all fail → queue → notify user → retry in 60s)"
        },
        "code": """
async function generateWithFallback(request) {
  const providers = [primaryProvider, backupProvider]
  for (const provider of providers) {
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        return await withTimeout(provider.generate(request), 30000)
      } catch (err) {
        if (attempt < 3) await sleep(attempt * 1000)
      }
    }
  }
  await jobQueue.add('generate', request, { delay: 60000 })
  return { status: 'queued', message: 'Content queued, you will be notified.' }
}
        """
    },
    "high_volume_architecture": {
        "components": [
            "API Gateway + Rate Limiter (per user/IP)",
            "Request Queue: SQS or BullMQ",
            "Worker Pool: Lambda or ECS (auto-scale on queue depth)",
            "Redis Cache: cache identical prompts (1hr TTL, saves 30-40% API cost)",
            "S3: output storage with CDN (CloudFront)",
            "WebSocket: notify user when job complete"
        ],
        "scaling_trigger": "Queue depth > 50 → scale out workers",
        "cost_optimization": "Cache hit for identical prompts → skip AI call entirely"
    }
}
