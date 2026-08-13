# Repository publication policy

Before publishing any branch, tag, release, pull request, source archive, or generated artifact to GitHub:

1. Treat raw conversations and facts derived from real conversations as private. This includes paraphrased fixtures, project names, paths, identifiers, commands, architecture details, metrics, URLs, and benchmark aggregates.
2. Use only deliberately fictional public fixtures. Do not anonymize a real example by changing only names or a few numbers.
3. Run `npm run release-check` with `YUGO_MEMORY_PRIVATE_DENYLIST` pointing to an untracked newline-delimited file of private phrases and identifiers.
4. Verify the working tree, every reachable Git blob, all branches and tags, PR/issue text, release notes, and generated source archives.
5. Never commit the private denylist, raw transcripts, memory databases, benchmark cases, or scanner output containing matched content.
6. If private material has already been published, do not rely on a follow-up deletion commit. Stop and obtain approval for history rewriting or repository recreation.

Installing or updating software remains subject to the user's explicit installation approval.
