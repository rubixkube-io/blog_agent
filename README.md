
# AI Blog Generation Pipeline

This project uses a **multi-agent architecture** powered by Google’s ADK (Agent Development Kit) to automatically generate professional, LinkedIn-ready blog posts.

## Overview

The system consists of coordinated AI agents that collaborate to plan, write, and compile a complete blog post from a single topic input.

### Agents Involved

* **`start_agent` (Root Agent):**
  Breaks down the main topic into three subtopics and delegates tasks to writer agents.

* **`writer_agent1`, `writer_agent2`, `writer_agent3`:**
  Work in parallel to generate the **introduction**, **main body**, and **conclusion** sections, respectively.

* **`aggregator_agent`:**
  Combines all sections into a cohesive, well-formatted Markdown article with sources and tags.

## ⚙️ Features

* Parallel content generation
* Structured and citation-aware writing
* Automatic compilation into a final publish-ready article
* Configurable tone, structure, and output schema

## Example Flow

1. The user provides a blog topic.
2. `start_agent` divides it into three subtopics.
3. Each writer agent generates content for one subtopic.
4. `aggregator_agent` merges all sections into the final blog.


---

Would you like me to make it a bit more formal (for GitHub) or more friendly and conversational (for internal use)?
