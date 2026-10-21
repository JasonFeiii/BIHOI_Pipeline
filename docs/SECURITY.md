# Security policy

Report vulnerabilities privately to the repository maintainers. Use GitHub private vulnerability reporting when enabled. Do not post passwords, tokens, private captures or calibration files in public issues. Security fixes target the current main branch.

## Deployment boundaries

- Bind to localhost by default. Network access requires authentication; do not expose the research WebUI directly to the Internet.
- Set `WEBUI_PASSWORD` through the environment or an ignored local `.env`. Configured and generated passwords are never printed. Generated credentials use owner-only runtime files.
- Configure `HOI_DATA_ROOT` and optional `HOI_ALLOWED_DATA_ROOTS` before launch. Browser inputs must stay within those roots after symlink resolution.
- Exports use fresh directories under the selected data root's `exports/`, or explicit `HOI_ALLOWED_EXPORT_ROOTS`. Export does not delete existing destination files.
- File deletion is a separate authenticated action requiring `DELETE`. Give the server account access only to necessary data.
- Obtain model files and native extensions from trusted sources. Serialized model formats can contain executable code.

These checks limit browser access; they do not isolate mutually untrusted users sharing an operating-system account. Use separate accounts or containers for that deployment.
