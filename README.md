# Trust Graph Engine

Multi-dimensional trust scoring with hermeneutic re-evaluation.

> Extracted from **Agent Friday** -- the AGI OS by [FutureSpeak.AI](https://futurespeak.ai).

## What is this?

The Trust Graph Engine models every person in your world with evidence-based credibility scoring across multiple domains. Each new observation triggers a hermeneutic re-evaluation -- the engine doesn't just append data, it re-interprets the entire picture.

### Key Concepts

- **Hermeneutic Circle** -- Every N observations (configurable), all trust dimensions are recomputed from all evidence, weighted by recency and impact. Between full re-evaluations, scores update incrementally.
- **Multi-Dimensional Trust** -- People are not simply "trusted" or "not trusted." Trust is scored across five core dimensions plus extensible custom dimensions.
- **Person Resolution** -- Fuzzy matching with Levenshtein distance, alias tracking (name, email, handle, phone, nickname), and first-name disambiguation. Mentions of "Jon Smith" and "John Smith" resolve to the same node.
- **Evidence-Based Scoring** -- Every trust change is backed by a typed evidence record with impact magnitude, timestamp, and optional domain tag.
- **Time Decay** -- Trust scores decay toward a floor over time without interaction, preventing stale high-trust entries.

## Installation

```bash
npm install @anthropic-ai/trust-graph-engine
```

## Quick Start

```typescript
import { TrustGraph } from '@anthropic-ai/trust-graph-engine';

const graph = new TrustGraph();

await graph.initialize({
  dataDir: './data/trust',
  config: {
    maxPersons: 200,
    evidenceRetention: 90,  // days
    decayRate: 0.001,
    reEvalThreshold: 5,
  },
});

// Resolve a person (creates if new, fuzzy-matches if existing)
const { person, isNew } = graph.resolvePerson('Sarah Chen');

// Add evidence
graph.addEvidence(person!.id, {
  type: 'promise_kept',
  description: 'Delivered the Q4 report on time as promised',
  impact: 0.8,
  domain: 'project-management',
});

graph.addEvidence(person!.id, {
  type: 'accurate_info',
  description: 'Market analysis predictions were correct',
  impact: 0.7,
  domain: 'market-research',
});

// Get context for system prompts or meeting prep
const context = graph.getContextForPerson(person!.id);
console.log(context);

// Get top-level prompt context for all key people
const promptContext = graph.getPromptContext();
```

## Trust Dimensions

| Dimension | Weight | Driven By |
|-----------|--------|-----------|
| **reliability** | 30% | `promise_kept` vs `promise_broken` evidence |
| **informationQuality** | 25% | `accurate_info` vs `inaccurate_info` evidence |
| **emotionalTrust** | 20% | `emotional_support` evidence (positive/negative impact) |
| **timeliness** | 15% | `helpful_action` vs `unhelpful_action` evidence |
| **expertise** | 10% | Per-domain scoring from domain-tagged evidence |

The `overall` score is a weighted composite of these dimensions. Each dimension ranges from 0 to 1, with 0.5 as the neutral starting point.

## Evidence Types

- `promise_kept` / `promise_broken` -- Commitment tracking
- `accurate_info` / `inaccurate_info` -- Information quality
- `helpful_action` / `unhelpful_action` -- Behavioral patterns
- `emotional_support` -- Emotional reliability
- `user_stated` -- Explicit user input about a person
- `observed` -- Inferred from conversation context
- `inferred` -- Derived from patterns

## API Overview

### Initialization
- `initialize(options)` -- Load or create trust graph with `dataDir` and optional `config`

### Person Resolution
- `resolvePerson(identifier, type?)` -- Resolve name/email/handle to a PersonNode (fuzzy match or create)
- `addAlias(personId, alias, type, confidence?)` -- Link an alias to an existing person

### Trust Management
- `addEvidence(personId, evidence)` -- Add trust evidence (triggers hermeneutic re-eval at threshold)
- `recomputeTrust(personId)` -- Force full hermeneutic re-evaluation
- `logCommunication(personId, event)` -- Log a communication event with sentiment

### Batch Processing
- `processPersonMentions(mentions)` -- Process person mentions from conversation extraction

### Context Generation
- `getContextForPerson(personId)` -- Full context string for a person
- `getContextForPersons(personIds)` -- Context for multiple people (meeting prep)
- `getPromptContext()` -- Top-level summary of key people for system prompts

### Queries
- `findByDomain(domain)` -- Find people by domain expertise
- `getMostTrusted(limit?)` -- Top trusted people
- `getRecentInteractions(limit?)` -- Most recently seen people
- `getAllPersons()` -- All persons sorted by last seen
- `getPersonById(id)` -- Lookup by ID
- `getPersonCount()` -- Total person count

### Person Management
- `updateNotes(personId, notes)` -- Set free-text notes on a person
- `linkPersons(personIdA, personIdB, label)` -- Create a bidirectional relationship

### Persistence
- `save()` -- Force a save to disk (auto-saves on mutations)

## Configuration

```typescript
interface TrustGraphConfig {
  maxPersons: number;        // Max tracked people (default: 200, evicts least active)
  evidenceRetention: number; // Days to keep evidence (default: 90)
  decayRate: number;         // Trust decay per day of no interaction (default: 0.001)
  reEvalThreshold: number;   // Evidence count triggering full hermeneutic re-eval (default: 5)
}
```

## License

MIT -- Copyright (c) 2025-2026 FutureSpeak.AI
