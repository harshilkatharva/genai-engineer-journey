# Memory Retention and User Controls

## Default retention

- **Working memory:** Held in process memory for the active conversation and removed
  when the conversation is cleared. Inactive entries expire after 24 hours.
- **Long-term memory:** Retained until the user deletes it or an explicit
  `expires_at` is set. Expired rows are excluded from retrieval and listing; a
  scheduled database cleanup should physically remove expired rows.
- **Episodic memory:** Retained for 90 days by default, then expires and is excluded
  from duplicate-action checks. A scheduled cleanup should physically remove it.

Retention durations are configurable. Deployments should schedule
`MemoryService.purge_expired()` to physically remove expired long-term and episodic
rows, and apply their database backup-retention and deletion policies to backups.

## User controls

Users may view their unexpired long-term memories, delete a specific memory, and
delete all memories through the memory service. Delete-all transactionally removes
the user's long-term and episodic rows, then clears that user's active working
memory from the current process. In a multi-process deployment, the application
must also invalidate working-memory entries in every process. Deletion from
backups follows the product's documented backup lifecycle.

## Sensitive information

Apply additional protections to personal identifiers and sensitive personal data:
minimize collection, restrict access, encrypt in transit and at rest, audit
administrative access, and use short retention where storage is necessary. Do not
extract or store passwords, API keys, access tokens, payment-card or bank-account
details, government identifiers, or sensitive health information as long-term
memory. Memory extraction is not a substitute for access control, encryption, or
user consent.

Never store authentication secrets, raw payment credentials, or private
conversation transcripts as long-term memory. Store only concise, useful facts,
preferences, goals, or constraints needed to improve future interactions.
