# /devblog - Post to Matt's devblog

Write and publish a devblog post focused on accomplishments and how colleagues can use what we've built.

## Arguments

$ARGUMENTS - The topic or feature to write about

## Instructions

You are writing a straightforward technical devblog post. Follow this process:

### 1. Check Repository Context

Run `git remote get-url origin` to check the repository:
- If it contains `github.com`, you can include a GitHub link
- If it contains `gogs` or is internal, you can include a Gogs link
- If not a git repo, discuss the work without linking

### 2. Draft the Post

Write a post that follows these principles:

**Focus: Accomplishments and Usage**
- What was built/accomplished
- How colleagues can use it
- Practical examples or use cases
- How AI assisted (show ROI on the Claude subscription)

**Tone: Mildly enthusiastic but not overboard**
- Professional and clear
- Helpful without being preachy
- Excited about what works, honest about limitations
- Direct and practical

**Structure:**
- Hook: The problem or opportunity
- What we built: Clear explanation
- How to use it: Practical steps
- AI involvement: Be specific about what Claude helped with
- Tags: Relevant technical tags

### 3. Show the Draft

Present the draft to the user with:
```
---
[Full post content]
---

Post this? [draft / publish / tweak]
```

### 4. Publish

If approved for publishing:

1. Read the API key from `~/.claude/devblog-api-key`
2. POST to `https://dw.ramsden-international.com/devblog/api/posts` with:

```json
{
  "title": "Post title here",
  "content": "Full markdown content",
  "repo": "GitHub or Gogs URL if applicable",
  "tags": ["relevant", "tags"],
  "publish": true
}
```

Include header: `X-Cyril-Key: <api-key>`

3. Report success with the post URL: `https://dw.ramsden-international.com/devblog/#/post/<slug>`

If saving as draft, set `"publish": false`.

### Example Topics

- "New natural language database query system deployed"
- "Automated reporting pipeline - how to use it"
- "Integration between X and Y systems"
- "Performance improvements in the data warehouse"

Always highlight how AI assisted in development or deployment.
