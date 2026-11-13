# cuco PR review checklist

A checklist of review rules for `cudax::cuco` (cuCollections inside NVIDIA/cccl), distilled from
the public review history of about 60 cuco-related NVIDIA/cccl pull requests.

It is written mainly for AI agents to self-review a cuco change before it is opened for human
review. Humans can use it too, but it is terse and organised for lookup, not for reading end to end.

## How to use it

- Point the agent at [`CUCO_PR_REVIEW_CHECKLIST.md`](CUCO_PR_REVIEW_CHECKLIST.md) and ask it to
  check the diff against the sections that apply (annotations, includes, noexcept, kernels, tests,
  CMake, and so on).
- Treat each item as a prompt to verify in the diff, not as an automatic comment. Check the claim
  against the code first and report only problems you have confirmed.
- Items marked **(recurring)** came up in two or more PRs. Items that say a review thread was
  unresolved or unadjudicated are open questions; do not cite them as agreed practice.

## Caveats

- Rules reflect cccl `main` as of the date at the top of the checklist and can go stale.
- Numbers like `#7705` refer to NVIDIA/cccl pull requests.
- It is a distillation, not an authoritative style guide; the cccl maintainers and the code in
  `main` take precedence.
