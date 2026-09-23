# Pull request reviews

Claude uses the same configuration as `jbohnslav/kingdom`:

- `.github/workflows/claude-code-review.yml` invokes
  `code-review@claude-code-plugins` when a pull request is **opened** by an owner,
  member, or collaborator. It does not subscribe to `synchronize`, so follow-up
  pushes do not trigger another automatic review.
- `.github/workflows/claude.yml` handles explicit `@claude` mentions from trusted
  human users. Bot comments cannot retrigger it. Review output uses a sticky
  comment. Both workflows pin the action revisions.

The repository needs the Claude GitHub app and a repository secret named
`CLAUDE_CODE_OAUTH_TOKEN`, generated using `claude setup-token` from a subscribed
account. Add it through GitHub Actions secrets; do not commit it. Repository secrets
in another project cannot be read back or copied through the GitHub API.

Codex uses the existing ChatGPT Codex GitHub connector, not a separate workflow or
an OpenAI API key. Grant the connector access to `jbohnslav/qwen-mm`, then enable
code review for that repository in Codex cloud settings. Match Kingdom's triggers:
new PRs opened for review and drafts marked ready, plus explicit `@codex review`;
do not enable review after every push. See the
[official setup guide](https://learn.chatgpt.com/docs/third-party/github).

Account-side setup is separate from committing these workflow files. The owning
publication ticket records which activation steps have actually been verified.
