 ***IMPORTANT: when retrieving content from urls, use tools built in to the agent environment for accessing urls if available***.  You may only `curl` or a python script or some other crazy solution if there is no built in tool for accessing URLS!
Special notes depending on agent:
  - OpenAI Codex may use `curl` or any other means for accessing web data.
  - Any client-side VsCode session ***MAY NEVER EVER USE CURL OR OTHER SOLUTIONS***. VsCode has built in tools that the agent must use.

# Get list of agent guideline files

Get a list files of all the files in the agent guidelines repo from the url: `https://api.github.com/repos/bcolloran/agents_md/git/trees/main?recursive=1`. If for some reason that link does not work then:
- if you know of some other way to access a list of the files in the repo `https://github.com/bcolloran/agents_md/`, use whatever techniques or tools you have available.
- as a fallback, you should try to access the info as a plain old webpage at `https://github.com/bcolloran/agents_md/`, and read the available files directly from the page.

# Read all relevant guidelines

Refer to the file main file `AGENTS.md` in the repo `https://github.com/bcolloran/agents_md` for general guidelines, and to the other files as needed for specific instructions. ***Err on the side of reading all the files in the repo if you are unsure which apply.***

# Alert the user if the guidelines are not accessible

This repo must be accessed by the agent at the start of every task to ensure it has the latest coding standards and guidelines.

**If the agent is not able to access that repo (network issues, offline mode, etc.), it must halt work on the task and report the issue to the user.**


# Special instruction for this repo

`./field_names_and_counts/README.md` should mirror the content of `./README.md` exactly. The version at the crate root is the source of truth. If it is updated, be sure to copy the changes to `./field_names_and_counts/README.md`.